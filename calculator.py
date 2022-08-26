import tkinter as tki
from PIL import Image, ImageTk
import cv2 as cv
from calculator_cv import CalculatorCV
from calculator_logic import CalculatorLogic

BUTTON_HOVER_COLOR = 'lightseagreen'
REGULAR_COLOR = 'cadetblue'
BUTTON_ACTIVE_COLOR = 'darkcyan'

BUTTON_STYLE = {"font": ("Courier", 30),
                "borderwidth": 1,
                "relief": tki.RAISED,
                "bg": REGULAR_COLOR,
                "activebackground": REGULAR_COLOR}


class Calculator:
    OPERATORS = {10: "*", 11: "/", 12: "=", 13: "+", 14: "-"}

    def __init__(self):
        self.__root = tki.Tk()
        self.__display_frame = tki.Frame(self.__root)
        self.__buttons_frame = tki.Frame(self.__root)
        self.__cam_frame = tki.Frame(self.__root)

        self.__display_frame.grid(row=0, column=0)
        self.__buttons_frame.grid(row=1, column=0)
        self.__cam_frame.grid(row=0, column=1, rowspan=2)

        self.__display_label = tki.Label(self.__display_frame, width=10, height=3,
                                         bg=BUTTON_HOVER_COLOR, font=("Courier", 30))
        self.__cam_label = tki.Label(self.__cam_frame, width=640, height=480)

        self.__display_label.pack()
        self.__buttons = {}
        self.create_buttons()
        self.__cam_label.pack()
        self.__logic = CalculatorLogic()
        self.__cv = CalculatorCV()
        self.show_frames([0, 0], True)

    def create_buttons(self):
        for i in range(4):
            tki.Grid.columnconfigure(self.__buttons_frame, i, weight=1)

        for i in range(5):
            tki.Grid.rowconfigure(self.__buttons_frame, i, weight=1)

        for i in range(10):
            row = 2-((i-1) // 3)
            col = (i-1) % 3 if i != 0 else 0
            self.create_single_button(str(i), row, col)
        self.create_single_button('*', 3, 1)
        self.create_single_button('/', 3, 2)
        self.create_single_button('+', 0, 3)
        self.create_single_button('-', 1, 3)
        self.create_single_button('=', 2, 3, rowspan=2)

    def create_single_button(self, text, row, col, rowspan=1):
        button = tki.Button(self.__buttons_frame, text=text, width=4, **BUTTON_STYLE)
        button.grid(row=row, column=col, rowspan=rowspan, sticky=tki.NSEW)
        self.__buttons[text] = button

    def press(self, char):
        button = self.__buttons[char]
        button["bg"] = BUTTON_ACTIVE_COLOR

        def done_press():
            button['bg'] = REGULAR_COLOR

        button.after(100, lambda: done_press())

    def update(self, img):
        image = ImageTk.PhotoImage(image=img)
        self.__cam_label.imgtk = image
        self.__cam_label.configure(image=image)

    def show_frames(self, location, down):
        im, location, down, min_key = self.__cv.film_and_compute(location, down)
        if min_key:
            print(min_key)
            if min_key < 10:
                button = str(min_key)
            else:
                button = self.OPERATORS[min_key]
            self.press(button)
            self.__logic.type_in(button)
        # Get the latest frame and convert into Image
        cv2image = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)

        # Convert image to PhotoImage
        imgtk = ImageTk.PhotoImage(image=img)
        self.__cam_label.imgtk = imgtk
        self.__cam_label.configure(image=imgtk)
        self.__display_label["text"] = self.__logic.get_display()
        self.__root.after(1, lambda: self.show_frames(location, down))

    def run(self):
        self.__root.mainloop()


if __name__ == '__main__':
    calc = Calculator()
    calc.run()
