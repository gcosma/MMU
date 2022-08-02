import tkinter
from tkinter import *
from tkinter.font import Font
from PIL import ImageTk, Image

import visualize


class MyGUI(object):
    def __init__(self):
        self.root = Tk()
        self.root.title('Test results visualised')
        self.custom_font = Font(family="Arial", size=13, weight="bold")

        # Fields
        self.lbl = Label(self.root, text="Query to view: (1-2000)")
        self.lbl.grid(column=0, row=0)
        self.lbl.configure(font=self.custom_font)
        self.txt = Entry(self.root, width=10)
        self.txt.grid(column=1, row=0)
        self.txt.configure(font=self.custom_font)
        self.button = Button(self.root, text="Result", command=self.on_button)
        self.button.grid(column=2, row=0)
        self.button.configure(font=self.custom_font)
        self.lblQuery = Label(self.root, text="Text query:")
        self.lblQuery.grid(column=3, row=0)
        self.lblQuery.configure(font=self.custom_font)

        self.lblCanvas1 = Label(self.root, text="")
        self.lblCanvas1.grid(column=0, row=1)
        self.lblCanvas1.configure(font=self.custom_font)
        self.lblCanvas2 = Label(self.root, text="")
        self.lblCanvas2.grid(column=1, row=1)
        self.lblCanvas2.configure(font=self.custom_font)
        self.lblCanvas3 = Label(self.root, text="")
        self.lblCanvas3.grid(column=2, row=1)
        self.lblCanvas3.configure(font=self.custom_font)
        self.lblCanvas4 = Label(self.root, text="")
        self.lblCanvas4.grid(column=3, row=1)
        self.lblCanvas4.configure(font=self.custom_font)
        self.lblCanvas5 = Label(self.root, text="")
        self.lblCanvas5.grid(column=0, row=3)
        self.lblCanvas5.configure(font=self.custom_font)
        self.lblCanvas6 = Label(self.root, text="")
        self.lblCanvas6.grid(column=1, row=3)
        self.lblCanvas6.configure(font=self.custom_font)
        self.lblCanvas7 = Label(self.root, text="")
        self.lblCanvas7.grid(column=2, row=3)
        self.lblCanvas7.configure(font=self.custom_font)
        self.lblCanvas8 = Label(self.root, text="")
        self.lblCanvas8.grid(column=3, row=3)
        self.lblCanvas8.configure(font=self.custom_font)

        self.canvas1 = Canvas(width=300, height=300, bg='white')
        self.canvas1.grid(column=0, row=2)
        self.canvas2 = Canvas(width=300, height=300, bg='white')
        self.canvas2.grid(column=1, row=2)
        self.canvas3 = Canvas(width=300, height=300, bg='white')
        self.canvas3.grid(column=2, row=2)
        self.canvas4 = Canvas(width=300, height=300, bg='white')
        self.canvas4.grid(column=3, row=2)
        self.canvas5 = Canvas(width=300, height=300, bg='white')
        self.canvas5.grid(column=0, row=4)
        self.canvas6 = Canvas(width=300, height=300, bg='white')
        self.canvas6.grid(column=1, row=4)
        self.canvas7 = Canvas(width=300, height=300, bg='white')
        self.canvas7.grid(column=2, row=4)
        self.canvas8 = Canvas(width=300, height=300, bg='white')
        self.canvas8.grid(column=3, row=4)

        # set first placeholders on canvas
        self.images = visualize.get_placeholders()
        self.photo1 = ImageTk.PhotoImage(self.images[0])
        self.img1 = self.canvas1.create_image(0, 0, image=self.photo1, anchor="nw")
        self.photo2 = ImageTk.PhotoImage(self.images[1])
        self.img2 = self.canvas2.create_image(0, 0, image=self.photo2, anchor="nw")
        self.photo3 = ImageTk.PhotoImage(self.images[2])
        self.img3 = self.canvas3.create_image(0, 0, image=self.photo3, anchor="nw")
        self.photo4 = ImageTk.PhotoImage(self.images[3])
        self.img4 = self.canvas4.create_image(0, 0, image=self.photo4, anchor="nw")
        self.photo5 = ImageTk.PhotoImage(self.images[4])
        self.img5 = self.canvas5.create_image(0, 0, image=self.photo5, anchor="nw")
        self.photo6 = ImageTk.PhotoImage(self.images[5])
        self.img6 = self.canvas6.create_image(0, 0, image=self.photo6, anchor="nw")
        self.photo7 = ImageTk.PhotoImage(self.images[6])
        self.img7 = self.canvas7.create_image(0, 0, image=self.photo7, anchor="nw")
        self.photo8 = ImageTk.PhotoImage(self.images[7])
        self.img8 = self.canvas8.create_image(0, 0, image=self.photo8, anchor="nw")

        self.root.mainloop()

    def on_button(self):
        query_number = int(self.txt.get())
        top_8_images, top_8_classes, top_8_results = visualize.give_top_image(query_number)

        self.lblCanvas1.configure(
            text="Retrieval 1 \n Classes: " + str(top_8_classes[0]) + "\n Correct retrieval: " + str(top_8_results[0]),
            font=self.custom_font)
        image1 = Image.open('rawdata/mirflickr/' + top_8_images[0])
        self.photo1 = ImageTk.PhotoImage(image1)
        self.canvas1.itemconfig(self.img1, image=self.photo1)
        self.lblCanvas2.configure(
            text="Retrieval 2 \n Classes: " + str(top_8_classes[1]) + "\n Correct retrieval: " + str(top_8_results[1]),
            font=self.custom_font)
        image2 = Image.open('rawdata/mirflickr/' + top_8_images[1])
        self.photo2 = ImageTk.PhotoImage(image2)
        self.canvas2.itemconfig(self.img2, image=self.photo2)
        self.lblCanvas3.configure(
            text="Retrieval 3 \n Classes: " + str(top_8_classes[2]) + "\n Correct retrieval: " + str(top_8_results[2]),
            font=self.custom_font)
        image3 = Image.open('rawdata/mirflickr/' + top_8_images[2])
        self.photo3 = ImageTk.PhotoImage(image3)
        self.canvas3.itemconfig(self.img3, image=self.photo3)
        self.lblCanvas4.configure(
            text="Retrieval 4 \n Classes: " + str(top_8_classes[3]) + "\n Correct retrieval: " + str(top_8_results[3]),
            font=self.custom_font)
        image4 = Image.open('rawdata/mirflickr/' + top_8_images[3])
        self.photo4 = ImageTk.PhotoImage(image4)
        self.canvas4.itemconfig(self.img4, image=self.photo4)
        self.lblCanvas5.configure(
            text="Retrieval 5 \n Classes: " + str(top_8_classes[4]) + "\n Correct retrieval: " + str(top_8_results[4]),
            font=self.custom_font)
        image5 = Image.open('rawdata/mirflickr/' + top_8_images[4])
        self.photo5 = ImageTk.PhotoImage(image5)
        self.canvas5.itemconfig(self.img5, image=self.photo5)
        self.lblCanvas6.configure(
            text="Retrieval 6 \n Classes: " + str(top_8_classes[5]) + "\n Correct retrieval: " + str(top_8_results[5]),
            font=self.custom_font)
        image6 = Image.open('rawdata/mirflickr/' + top_8_images[5])
        self.photo6 = ImageTk.PhotoImage(image6)
        self.canvas6.itemconfig(self.img6, image=self.photo6)
        self.lblCanvas7.configure(
            text="Retrieval 7 \n Classes: " + str(top_8_classes[6]) + "\n Correct retrieval: " + str(top_8_results[6]),
            font=self.custom_font)
        image7 = Image.open('rawdata/mirflickr/' + top_8_images[6])
        self.photo7 = ImageTk.PhotoImage(image7)
        self.canvas7.itemconfig(self.img7, image=self.photo7)
        self.lblCanvas8.configure(
            text="Retrieval 8 \n Classes: " + str(top_8_classes[7]) + "\n Correct retrieval: " + str(top_8_results[7]),
            font=self.custom_font)
        image8 = Image.open('rawdata/mirflickr/' + top_8_images[7])
        self.photo8 = ImageTk.PhotoImage(image8)
        self.canvas8.itemconfig(self.img8, image=self.photo8)

        query_tags, query_classes = visualize.give_query(query_number)
        query_display_text = "Query classes:" + str(query_classes) + "\n" + query_tags
        self.lblQuery.configure(text=query_display_text)


def run_gui():
    gui = MyGUI()


if __name__ == "__main__":
    gui = MyGUI()
