import tkinter
from tkinter import *
from tkinter.font import Font
from PIL import ImageTk, Image
import numpy as np
from scipy import io

import visualize


class MyGUI(object):
    def __init__(self):
        self.position = 0
        self.subclasses = []

        self.root = Tk()
        self.root.title('Relabel dataset')
        self.custom_font = Font(family="Arial", size=13, weight="bold")
        self.root.bind('<KeyPress>', self.onKeyPress)

        # Fields
        self.lbl = Label(self.root, text="1:Dog, 2:Cat, 3:Horse, 4:Fish, 5:Bird, 6:Rabbit, 7:Insect, 8:Crustacean, "
                                         "9:Inanimate animal representation, 0:Other \n -:Back")
        self.lbl.grid(column=0, row=0, columnspan=2)
        self.canvas = Canvas(width=480, height=480, bg='white')
        self.canvas.grid(column=0, row=1)
        self.lbl2 = Label(self.root, text="")
        self.lbl2.grid(column=1, row=1)
        self.lbl3 = Label(self.root, text="")
        self.lbl3.grid(column=0, row=2, columnspan=2)

        # set first placeholders on canvas
        starter, _ = visualize.scroll_class1(self.position)
        image = Image.open(starter)
        self.photo = ImageTk.PhotoImage(image)
        self.img = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.nextImage()

        self.root.mainloop()

        # [Dog, Cat, Horse, Fish, Bird, Rabbit, Insect, Crustacean, Inanimate animal representation, Other]

    def onKeyPress(self, event):
        # print(self.position)
        if self.position < 2578:
            if event.char == "1":
                self.subclasses.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                self.nextImage()
                self.lbl3.configure(text="Previous image set to: 1:Dog")
            if event.char == "2":
                self.subclasses.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
                self.nextImage()
                self.lbl3.configure(text="Previous image set to: 2:Cat")
            if event.char == "3":
                self.subclasses.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
                self.nextImage()
                self.lbl3.configure(text="Previous image set to: 3:Horse")
            if event.char == "4":
                self.subclasses.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
                self.nextImage()
                self.lbl3.configure(text="Previous image set to: 4:Fish")
            if event.char == "5":
                self.subclasses.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
                self.nextImage()
                self.lbl3.configure(text="Previous image set to: 5:Bird")
            if event.char == "6":
                self.subclasses.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
                self.nextImage()
                self.lbl3.configure(text="Previous image set to: 6:Rabbit")
            if event.char == "7":
                self.subclasses.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
                self.nextImage()
                self.lbl3.configure(text="Previous image set to: 7:Insect")
            if event.char == "8":
                self.subclasses.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
                self.nextImage()
                self.lbl3.configure(text="Previous image set to: 8:Crustacean")
            if event.char == "9":
                self.subclasses.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
                self.nextImage()
                self.lbl3.configure(text="Previous image set to: 9:Inanimate animal representation")
            if event.char == "0":
                self.subclasses.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
                self.nextImage()
                self.lbl3.configure(text="Previous image set to: 10:Other")
            elif event.char == "a" and self.position > 0:
                self.position -= 1
                image, text = visualize.scroll_class1(self.position)
                image_change = Image.open(image)
                self.photo = ImageTk.PhotoImage(image_change)
                self.canvas.itemconfig(self.img, image=self.photo)
                del self.subclasses[-1]
        elif self.position == 2578:
            print("Labelling completed")
            npsubclasses = np.asarray(self.subclasses, dtype=int)
            data = {'LAllClassOne': npsubclasses}
            io.savemat('LAllClassOne.mat', data)

    def nextImage(self):
        self.position += 1
        image, text = visualize.scroll_class1(self.position)
        image_change = Image.open(image)
        self.photo = ImageTk.PhotoImage(image_change)
        self.canvas.itemconfig(self.img, image=self.photo)
        self.lbl2.configure(text=text)


def run_gui():
    gui = MyGUI()


if __name__ == "__main__":
    gui = MyGUI()
