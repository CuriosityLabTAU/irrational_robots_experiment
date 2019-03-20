# The code for changing pages was derived from: http://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter
# License: http://creativecommons.org/licenses/by-sa/3.0/

# import tkinter as tk # python 3
import Tkinter as tk
import numpy as np

from pygame import mixer
mixer.init(frequency=16000, size=-16, channels=2, buffer=2048)

from PIL import Image, ImageTk
import random
from time import sleep

LARGE_FONT = ("Verdana", 12)
# path = 'desktop_app/images/'
path = 'images/'

# sounds_path = 'desktop_app/sounds/'
sounds_path = 'sounds/'

import threading
Gender = 'f'

class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry('1050x600')  # set size of the main window to 300x300 pixels
        container = tk.Frame(self, background="black")

        container.pack(side="top", fill="both", expand=True)
        self.gender = None
        self.frames = {}

        for F in (StartPage, ControlScreen,OpeningPage, PageOne, PageTwo, PageThree, PageFour, PageFive, PageSix,PageSeven, EndPage):
            frame = F(container, self)

            self.frames[F] = frame
            frame.config(bg='black')  # change the background color to black
            frame.grid(row=3, column=3, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        for i, F in enumerate((ControlScreen, OpeningPage, PageOne, PageTwo, PageThree, PageFour, PageFive, PageSix, PageSeven, EndPage)):
            b = tk.Button(self, text='visit Page %d' % (i + 1),command=lambda f=F: controller.show_frame(f))
            b.pack()

def transition(widget_values, controller, page, gender = 'f', parent = None):
    global Gender

    if page == OpeningPage:
        Gender = widget_values['gender'].get()


    if widget_values != None:
        d = {}
        try: # when we have ratings (in a dict)
            for k, val in widget_values.items():
                v = val.get()
                if type(v) == 'str':
                    v = int(v)
                d[k] = v
            print(d)
        except:
            print(widget_values)

    controller.show_frame(page)

    t = threading.Thread(target = play_file, args = (page, Gender,))
    t.start()

    # play_file(page, gender)

def agree_with(self1, controller, page, n = 2):
    '''creating agree screen'''

    image = Image.open(path + 'robots_finshed_talking' + '.png')
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(self1, image=photo, bg='black')
    label.image = photo  # keep a reference!
    label.grid(row=n, columnspan=10, sticky='e')

    image = Image.open(path + 'agree' + '.png')
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(self1, image=photo, bg='black')
    label.image = photo  # keep a reference!
    label.grid(row=n+1, columnspan=10, sticky='e', pady = 10)

    image = Image.open(path + 'red' + '.png')
    photo = ImageTk.PhotoImage(image)
    clr = '#%02x%02x%02x' % (255, 80, 80)
    red_button = tk.Button(self1, image=photo, bg = clr,
                           command=lambda: transition('red', controller, page))
    red_button.image = photo
    red_button.grid(row=n + 2, column=1, pady = 30)

    image = Image.open(path + 'blue' + '.png')
    photo = ImageTk.PhotoImage(image)
    clr = '#%02x%02x%02x' % (47, 85, 151)
    blue_button = tk.Button(self1, image=photo, bg = clr,
                            command=lambda: transition('blue', controller, page))
    blue_button.image = photo
    blue_button.grid(row=n + 2, column=8, pady = 30)

def pleas_rate(self, suspects):
    '''creating rating options'''

    ### randomize the order that the rating options are presented
    random.shuffle(suspects)

    scales = {}

    for i, photo in enumerate(suspects):
        scales[photo] = tk.Scale(self, from_=100, to=0, orient='horizontal', resolution=10, length=350, bg='black', fg='white')

        scales[photo].config(highlightthickness=0)
        scales[photo].grid(row=i + 2, column=1, columnspan=7, padx=10, pady=20, sticky='n')

        image = Image.open(path + 'suspect_' + photo + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=i + 2, column=9, sticky='e')

        likely = tk.Label(self, text = '100 %', bg='black', fg='white')
        likely.grid(row=i + 2, column=0, sticky='e')

        not_likely = tk.Label(self, text='0 %', bg='black', fg='white')
        not_likely.grid(row=i + 2, column=8, sticky='e')

    return scales, i + 2

def next_button(self, scales, controller, page, i, gender = 'f', parent = None):
    button1 = tk.Button(self, text="<--", width=20,
                        command=lambda: transition(scales, controller, page, gender, parent))
    button1.grid(row=i + 1, column=1, columnspan=2)



class ControlScreen(tk.Frame):
    global Gender
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Control Page", font=LARGE_FONT)
        label.grid(row=0, columnspan =2)

        enteries = {}
        for i, txt in enumerate(['user id', 'gender']):
            label = tk.Label(self, text = txt.capitalize(), bg='black', fg='white')
            label.grid(row=i+1, column =0, sticky='e')
            enteries[txt] = tk.Entry(self)
            enteries[txt].grid(row=i+1, column = 1)

        Gender = enteries['gender'].get().strip().lower()

        next_button(self, enteries, controller, OpeningPage, 10)


class PageOne(tk.Frame):
    global Gender
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        image = Image.open(path + 'story_1' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=0, columnspan=10, sticky='e')
        print(parent.master.gender)

        # if parent.master.gender == 'm': # todo: insert gender to parent somehow
        if Gender == 'm':
            print('male!')


        image = Image.open(path + 'rate' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=1, columnspan = 10, sticky='e')


        scales, i = pleas_rate(self, ['a', 'b', 'a_and_b'])

        next_button(self, scales, controller, PageTwo, i)

def play_file(page = None, gender = 'f'):
    d = []
    if page == PageOne:
        d.append(mixer.Sound(sounds_path + 'intor0_1.wav'))
        d.append(mixer.Sound(sounds_path + 'intro0_2.wav'))
        if gender == 'm':
            d.append(mixer.Sound(sounds_path + 'intro0_3m.wav'))
        else:
            d.append(mixer.Sound(sounds_path + 'intro0_3f.wav'))
        d.append(mixer.Sound(sounds_path + 'rate.wav'))
        for i in d:
            while mixer.get_busy():
                pass
            i.play()

class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        image = Image.open(path + 'story_2' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=0, columnspan=10, sticky='e')

        agree_with(self, controller, PageThree, n = 2)


class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        image = Image.open(path + 'rate' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=1, columnspan = 10, sticky='e')

        scales, i = pleas_rate(self,['a', 'c', 'a_and_c'])

        next_button(self, scales, controller, PageFour, i)


class PageFour(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        image = Image.open(path + 'story_3' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=0, columnspan=10, sticky='e')

        agree_with(self, controller, PageFive, n = 2)

class PageFive(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        image = Image.open(path + 'robots_finshed_talking' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=0, columnspan = 4, sticky='e')

        image = Image.open(path + 'please_rank' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=1, columnspan=4, sticky='e')

        rankings = {}
        w,h = 401, 46
        for i, photo in enumerate(['a','b', 'c', 'd']):
            rankings[photo] = tk.Entry(self,  width = 5)
            rankings[photo].grid(row=i + 2, column=2, pady=0, sticky='n')

            image = Image.open(path + 'suspect_' + photo + '.png')
            # w, h = image.size
            image = image.resize((w, h), Image.ANTIALIAS)  # The (250, 250) is (height, width)
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(self, image=photo, bg='black')
            label.image = photo  # keep a reference!
            label.grid(row=i + 2, column=3, sticky='e', pady=0)

        for i, photo in enumerate(['a_and_b', 'a_and_c', 'a_and_d', 'b_and_c', 'b_and_d', 'c_and_d']):
            rankings[photo] = tk.Entry(self, width=5)
            rankings[photo].grid(row=i + 2, column=0, pady=0, sticky='n')

            image = Image.open(path + 'suspect_' + photo + '.png')
            image = image.resize((w, h), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(self, image=photo, bg='black')
            label.image = photo  # keep a reference!
            label.grid(row=i + 2, column=1, sticky='s', padx=0)

        # button1 = tk.Button(self, text="<--", width=20,
        #                     command=lambda: controller.show_frame(PageSix))
        # button1.grid(row=i + 2, column=1, columnspan=1)

        next_button(self, rankings, controller, PageSix, i+2)


class PageSix(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        image = Image.open(path + 'robots_finshed_talking' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=0, columnspan = 2, sticky='e')

        image = Image.open(path + 'who_did_it' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=1, columnspan=2, sticky='e')

        rankings = {}
        w,h = 401, 46
        for i, p in enumerate(['a','b', 'c', 'd']):
            image = Image.open(path + 'suspect_' + p + '.png')
            image = image.resize((w, h), Image.ANTIALIAS)  # The (250, 250) is (height, width)
            photo = ImageTk.PhotoImage(image)
            rankings[p] = tk.Button(self, image=photo, bg='black',command=lambda p=p: transition([p], controller, PageSeven))
            rankings[p].image = photo  # keep a reference!
            rankings[p].grid(row=i + 2, column=1, sticky='s', padx=0, pady = 5)

        for i, p in enumerate(['a_and_b', 'a_and_c', 'a_and_d', 'b_and_c', 'b_and_d', 'c_and_d']):
            image = Image.open(path + 'suspect_' + p + '.png')
            image = image.resize((w, h), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            rankings[p] = tk.Button(self, image=photo, bg='black',command=lambda p=p: transition([p], controller, PageSeven))
            rankings[p].image = photo  # keep a reference!
            rankings[p].grid(row=i + 2, column=0, sticky='s', padx=0, pady = 5)

class PageSeven(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        n = -1
        image = Image.open(path + 'hire_detectivev' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=n + 1, columnspan=10, sticky='e', pady=10)

        image = Image.open(path + 'red' + '.png')
        photo = ImageTk.PhotoImage(image)
        clr = '#%02x%02x%02x' % (255, 80, 80)
        red_button = tk.Button(self, image=photo, bg=clr,
                               command=lambda: transition(['red'], controller, EndPage))
        red_button.image = photo
        red_button.grid(row=n + 2, column=1, pady=30)

        image = Image.open(path + 'blue' + '.png')
        photo = ImageTk.PhotoImage(image)
        clr = '#%02x%02x%02x' % (47, 85, 151)
        blue_button = tk.Button(self, image=photo, bg=clr,
                                command=lambda: transition(['blue'], controller, EndPage))
        blue_button.image = photo
        blue_button.grid(row=n + 2, column=8, pady=30)


class EndPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)


        image = Image.open(path + 'the_end' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=0, sticky='e', pady=10)

        clr = '#%02x%02x%02x' % (146, 208, 80)
        image = Image.open(path + 'end_button' + '.png')
        photo = ImageTk.PhotoImage(image)

        b = tk.Button(self, image=photo, bg=clr, command=self.quit)
        b.grid(row=1, pady=10)
        b.image = photo


class OpeningPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)


        image = Image.open(path + 'begin_text' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=0, sticky='e', pady=10)

        clr = '#%02x%02x%02x' % (146, 208, 80)
        image = Image.open(path + 'begin_button' + '.png')
        photo = ImageTk.PhotoImage(image)

        b = tk.Button(self, image=photo, bg=clr, command=lambda: transition(None, controller, PageOne, parent.master.gender))
        b.grid(row=1, pady=10)
        b.image = photo

app = SeaofBTCapp()
app.mainloop()

# relative positioning --> independent of the screen size --> better for full screen
# https://www.python-course.eu/tkinter_layout_management.phpv --> see *.place()
# todo: torr remember you need it to work not to be the prettiest

# todo: sound files!!!