import torch
from PIL import Image
import numpy as np
import visdom
import time


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def calc_map_k(qB, rB, query_label, retrieval_label, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # sim: {0, 1}^{mxn}
    num_query = query_label.shape[0]
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map


def calc_map_k_classes(qB, rB, query_label, retrieval_label, k=None):
    num_query = query_label.shape[0]
    label_check = query_label.cpu()
    label_check = label_check.tolist()
    map = 0.
    map_classes = []
    map_classes_count = []
    for i in label_check[0]:
        map_classes.append(0.)
        map_classes_count.append(0)
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0

        new_label_check = [int(o) for o in label_check[i]]
        for e in range(0, len(label_check[i])):
            if new_label_check[int(e)] == 1:
                map_classes[e] += torch.mean(count / tindex)
                map_classes_count[e] += 1
        map += torch.mean(count / tindex)
    for i in range(0, len(map_classes_count)):
        print(float(map_classes[i] / map_classes_count[i]))
        #print('Class', i+1, "(" + str(map_classes_count[i]) + ")")
        map_classes[i] = map_classes[i] / map_classes_count[i]
    map = map / num_query
    return map


def image_from_numpy(x):
    if x.max() > 1.0:
        x = x / 255
    if type(x) != np.ndarray:
        x = x.numpy()
    im = Image.fromarray(np.uint8(x * 255))
    im.show()


def pr_curve(qB, rB, query_label, retrieval_label):
    num_query = qB.shape[0]
    num_bit = qB.shape[1]
    P = torch.zeros(num_query, num_bit + 1)
    R = torch.zeros(num_query, num_bit + 1)
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask
    return P, R


def p_topK(qB, rB, query_label, retrieval_label, K):
    num_query = query_label.shape[0]
    p = [0] * len(K)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm)[1][:total]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p


class Visualizer(object):
    """
    ?????????visdom?????????????????????????????????????????????`self.vis.function`
    ???????????????visdom??????
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        # ???????????????????????????????????????
        # ????????????loss',23??? ???loss??????23??????
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        ??????visdom?????????
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        ??????plot??????
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name, opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        ?????????don???t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~?????????
        """
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)


if __name__ == '__main__':
    qB = torch.Tensor([[1, -1, 1, 1],
                       [-1, -1, -1, 1],
                       [1, 1, -1, 1],
                       [1, 1, 1, -1]])
    rB = torch.Tensor([[1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [1, 1, -1, -1],
                       [-1, 1, -1, -1],
                       [1, 1, -1, 1]])
    query_labels = torch.Tensor([[0, 1, 0, 0],
                                 [1, 1, 0, 0],
                                 [1, 0, 0, 1],
                                 [0, 1, 0, 1]])
    retrieval_labels = torch.Tensor([[1, 0, 0, 1],
                                     [1, 1, 0, 0],
                                     [0, 1, 1, 0],
                                     [0, 0, 1, 0],
                                     [1, 0, 0, 0],
                                     [0, 0, 1, 0]])

    # query_labels = torch.Tensor([[0, 1, 0, 0],
    #                              [1, 0, 0, 0],
    #                              [1, 0, 0, 0],
    #                              [0, 1, 0, 0]])
    # retrieval_labels = torch.Tensor([[1, 0, 0, 0],
    #                                  [0, 1, 0, 0],
    #                                  [0, 0, 1, 0],
    #                                  [0, 0, 1, 0],
    #                                  [1, 0, 0, 0],
    #                                  [0, 0, 1, 0]])

    trn_bainary = torch.Tensor(
        [
            [1, -1, 1, 1, -1],
            [-1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1],
            [-1, 1, 1, -1, 1]
        ]
    )
    tst_binary = torch.Tensor(
        [
            [1, 1, 1, -1, -1],
            [1, 1, 1, 1, 1]
        ]
    )
    trn_label = torch.Tensor(
        [[0, 1], [0, 1], [1, 0], [1, 0]]
    )
    tst_label = torch.Tensor(
        [[1, 0], [0, 0]]
    )

    map = calc_map_k(qB, rB, query_labels, retrieval_labels)
    # map = calc_map(tst_binary, trn_bainary, tst_label, trn_label)
    print(map)
    # a = torch.randint(0, 256, (224, 224, 3))
    # image_from_numpy(a)
