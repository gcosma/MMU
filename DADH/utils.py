import torch
import numpy as np
import visdom
from scipy import io


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def calc_map_k(qB, rB, query_label, retrieval_label, k=None):
    num_query = query_label.shape[0]
    map = 0.
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
        map += torch.mean(count / tindex)
    map = map / num_query
    return map


def calc_map_k_custom_class1(qB, rB, query_label, retrieval_label, class1_query_label, class1_retrieval_label, k=None):
    num_query = query_label.shape[0]
    map = 0.
    class1_map = 0.
    class1_num_query = 0
    skipcount = 0
    subclass_map = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    subclass_map_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
        map += torch.mean(count / tindex)
        if query_label[i][0] == 1:
            gnd_c1 = (class1_query_label[i].unsqueeze(0).mm(class1_retrieval_label.t()) > 0).type(torch.float).squeeze()
            tsum = torch.sum(gnd_c1)
            if tsum == 0 or class1_query_label[i][8] == 1 or class1_query_label[i][9] == 1:
                skipcount += 1
                continue
            hamm = calc_hamming_dist(qB[i, :], rB)
            _, ind = torch.sort(hamm)
            ind.squeeze_()
            gnd_c1 = gnd_c1[ind]
            total = min(k, int(tsum))
            count = torch.arange(1, total + 1).type(torch.float).to(gnd_c1.device)
            tindex = torch.nonzero(gnd_c1)[:total].squeeze().type(torch.float) + 1.0
            class1_map += torch.mean(count / tindex)
            class1_num_query += 1

            label_check = class1_query_label.cpu()
            label_check = label_check.tolist()
            new_label_check = [int(o) for o in label_check[i]]
            for e in range(0, len(new_label_check)):
                if int(new_label_check[e]) == 1:
                    subclass_map_count[e] += 1
                    subclass_map[e] += torch.mean(count / tindex)
    map = map / num_query
    class1_map = class1_map / class1_num_query
    for i in range(0, len(subclass_map_count)):
        if subclass_map_count[i] == 0:
            subclass_map[i] = "N/A"
        else:
            subclass_map[i] = float(subclass_map[i] / subclass_map_count[i])
        print(subclass_map[i])
    print("skipcount:", skipcount)
    return map, class1_map


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
        map_classes[i] = map_classes[i] / map_classes_count[i]
    map = map / num_query
    return map


def calc_map_k_ind_extract(qB, rB, query_label, retrieval_label, k=None):
    num_query = query_label.shape[0]
    map = 0.
    indices = []
    results = []
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)

        indices.append(ind.cpu().tolist())

        ind.squeeze_()
        gnd = gnd[ind]

        results.append(gnd.cpu().tolist())

        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)

    npindices = np.asarray(indices, dtype=int)
    npresults = np.asarray(results, dtype=int)
    data = {'indices': npindices, 'results': npresults}
    io.savemat('results.mat', data)

    map = map / num_query
    return map


class Visualizer(object):

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        self.index = {}

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name, opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def __getattr__(self, name):
        return getattr(self.vis, name)
