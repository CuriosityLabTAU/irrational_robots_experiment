# The code for changing pages was derived from: http://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter
# License: http://creativecommons.org/licenses/by-sa/3.0/

# import tkinter as tk # python 3
import Tkinter as tk
import numpy as np

from pygame import mixer
# mixer.init(frequency=16000, size=-16, channels=2, buffer=2048)
import pygame
pygame.init()


from PIL import Image, ImageTk
import random
from time import sleep

LARGE_FONT = ("Verdana", 12)

import threading

Gender = 'f'
# Gender = 'm'

path, sounds_path = 'images/', 'sounds/'
first_story = 'suspect'

class SeaofBTCapp(tk.Tk):

    def __init__(self, test = False, *args, **kwargs):
        global path, sounds_path
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry('1050x700')  # set size of the main window to 300x300 pixels
        if test == False:
            sounds_path = 'desktop_app/sounds/'
            path = 'desktop_app/images/'
        container = tk.Frame(self, background="black")

        container.pack(side="top", fill="both", expand=True)
        self.gender = None
        self.frames = {}

        for F in (StartPage, ControlScreen,OpeningPage, PageOne, PageTwo, PageThree, PageFour, PageFive, PageSix,PageSeven,
                  artOne, artTwo, artThree, artFour, artFive, artSix,artSeven,
                  case2, EndPage):
            frame = F(container, self)

            self.frames[F] = frame
            frame.config(bg='black')  # change the background color to black
            frame.grid(row=3, column=3, sticky="nsew")

        # self.show_frame(StartPage)
        self.show_frame(ControlScreen)

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

def transition(widget_values, controller, page, gender = 'f', parent = None, p = None):
    global Gender, first_story
    if p == 'case2':
        p = list(set(['suspect', 'art']) - set([first_story]))[0]

    if page == OpeningPage:
        Gender = widget_values['gender'].get()
        first_story = widget_values['first story'].get()

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

    if first_story == 'suspect':
        if p == 'suspect':
            if page == EndPage:
                page = case2
                print('starting story 2')
        elif p == 'art':
            if page == EndPage:
                pass
            else:
                page = page[1]
        try:
            controller.show_frame(page[0])
            page = page[0]
        except:
            controller.show_frame(page)

    elif first_story == 'art':
        if p == 'art':
            if page == EndPage:
                page = case2
                print('starting story 2')
        elif p == 'suspect':
            if page == EndPage:
                pass
            else:
                page = page[0]

        try:
            controller.show_frame(page[1])
            page = page[1]
        except:
            controller.show_frame(page)


    if 'Page' in page.__name__: # and page == PageOne:
        t = threading.Thread(target = play_file_suspects, args = (page, Gender,))
        t.start()
    elif 'art' in page.__name__:
        t = threading.Thread(target = play_file_art, args=(page, Gender,))
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

def pleas_rate_suspects(self, suspects):
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

def pleas_rate_art(self, suspects):
    '''creating rating options'''

    ### randomize the order that the rating options are presented
    random.shuffle(suspects)

    scales = {}

    for i, photo in enumerate(suspects):
        scales[photo] = tk.Scale(self, from_=100, to=0, orient='horizontal', resolution=10, length=350, bg='black', fg='white')

        scales[photo].config(highlightthickness=0)
        scales[photo].grid(row=i + 2, column=1, columnspan=7, padx=10, pady=20, sticky='n')

        image = Image.open(path + 'art_' + photo + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=i + 2, column=9, sticky='e')

        likely = tk.Label(self, text = '100 %', bg='black', fg='white')
        likely.grid(row=i + 2, column=0, sticky='e')

        not_likely = tk.Label(self, text='0 %', bg='black', fg='white')
        not_likely.grid(row=i + 2, column=8, sticky='e')

    return scales, i + 2

def next_button(self, scales, controller, page, i, gender = Gender, parent = None):
    button1 = tk.Button(self, text="<--", width=20,
                        command=lambda: transition(scales, controller, page, gender, parent))
    button1.grid(row=i + 1, column=1, columnspan=2)



class ControlScreen(tk.Frame):
    def __init__(self, parent, controller):
        global Gender, first_story
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Control Page", font=LARGE_FONT)
        label.grid(row=0, columnspan =2)

        enteries = {}
        def_vals = ['000', Gender, 'right', 'rational', 'left', 'irrational', 'suspect']
        for i, txt in enumerate(['user id', 'gender', 'red side', 'red rationality', 'blue side', 'blue rationality', 'first story']):
            label = tk.Label(self, text = txt.capitalize(), bg='black', fg='white')
            label.grid(row=i+1, column =0, sticky='e')

            v = tk.StringVar(self, value=def_vals[i])

            enteries[txt] = tk.Entry(self, textvariable=v)
            enteries[txt].grid(row=i + 1, column=1)

        Gender = enteries['gender'].get().strip().lower()

        next_button(self, enteries, controller, OpeningPage, 10)


class PageOne(tk.Frame):
    global Gender
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        image = Image.open(path + 'diamonds_intro_%s' % Gender + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=0, columnspan=10, sticky='e')
        # print(parent.master.gender)

        # # if parent.master.gender == 'm': # todo: insert gender to parent somehow
        # if Gender == 'm':
        #     print('male!')


        image = Image.open(path + 'rate.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=1, columnspan = 10, sticky='e')


        scales, i = pleas_rate_suspects(self, ['a', 'b', 'a_and_b'])

        next_button(self, scales, controller, PageTwo, i)

def play_file_suspects(page = None, gender ='f'):
    d = []
    if page == OpeningPage:
        d.append(mixer.Sound(sounds_path + 'opening_%s.wav' % gender))
    elif page == PageOne:
        d.append(mixer.Sound(sounds_path + 'diamonds_intro_0_%s.wav' % gender))
        d.append(mixer.Sound(sounds_path + 'diamonds_intro_1.wav'))
        d.append(mixer.Sound(sounds_path + 'intro0_3%s.wav' % gender))
        d.append(mixer.Sound(sounds_path + 'rate_suspects.wav'))
    elif page == PageTwo:
        d.append(mixer.Sound(sounds_path + 'suspects_c_d.wav'))
    elif page == PageThree:
        d.append(mixer.Sound(sounds_path + 'rate_suspects.wav'))
    elif page == PageFour:
        d.append(mixer.Sound(sounds_path + 'suspect_b_d.wav'))
    elif page == PageFive:
        d.append(mixer.Sound(sounds_path + 'rate_all_10_%s.wav' % gender))
    elif page == PageSix:
        d.append(mixer.Sound(sounds_path + 'who_did_it.wav'))
    elif page == PageSeven:
        d.append(mixer.Sound(sounds_path + 'robot_detective_%s.wav' % gender))
    for i in d:
        while mixer.get_busy():
            pass
        i.play()

def play_file_art(page=None, gender='f'):
    d = []
    if page == artOne:
        d.append(mixer.Sound(sounds_path + 'art_intro_0.wav'))
        d.append(mixer.Sound(sounds_path + 'art_intro_1_%s.wav'% gender))
        d.append(mixer.Sound(sounds_path + 'rate_art.wav'))
    elif page == artTwo:
        d.append(mixer.Sound(sounds_path + 'caught_young.wav'))
    elif page == artThree:
        d.append(mixer.Sound(sounds_path + 'rate_art.wav'))
    elif page == artFour:
        d.append(mixer.Sound(sounds_path + 'most_expensive.wav'))
    elif page == artFive:
        d.append(mixer.Sound(sounds_path + 'rank_all_9.wav'))
    elif page == artSix:
        d.append(mixer.Sound(sounds_path + 'what_was_most_expensive_piece.wav'))
    elif page == artSeven:
        d.append(mixer.Sound(sounds_path + 'robot_art_buyer_%s.wav' % gender))
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

        scales, i = pleas_rate_suspects(self, ['a', 'c', 'a_and_c'])

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
        #
        # image = Image.open(path + 'robots_finshed_talking' + '.png')
        # photo = ImageTk.PhotoImage(image)
        # label = tk.Label(self, image=photo, bg='black')
        # label.image = photo  # keep a reference!
        # label.grid(row=0, columnspan = 4, sticky='e')

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
        global Gender
        tk.Frame.__init__(self, parent)
        n = -1
        image = Image.open(path + 'hire_detective_%s' % Gender+ '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=n + 1, columnspan=10, sticky='e', pady=10)

        image = Image.open(path + 'red' + '.png')
        photo = ImageTk.PhotoImage(image)
        clr = '#%02x%02x%02x' % (255, 80, 80)
        red_button = tk.Button(self, image=photo, bg=clr,
                               command=lambda: transition(['red'], controller, EndPage, p = 'suspect'))
        red_button.image = photo
        red_button.grid(row=n + 2, column=1, pady=30)

        image = Image.open(path + 'blue' + '.png')
        photo = ImageTk.PhotoImage(image)
        clr = '#%02x%02x%02x' % (47, 85, 151)
        blue_button = tk.Button(self, image=photo, bg=clr,
                                command=lambda: transition(['blue'], controller, EndPage, p = 'suspect'))
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


class artOne(tk.Frame):
    global Gender
    # print(Gender)


    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        image = Image.open(path + 'painting_intro_%s.png' % Gender)
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=0, columnspan=10, sticky='e')
        # print(parent.master.gender)

        # image = Image.open(path + 'rate' + '.png')
        # photo = ImageTk.PhotoImage(image)
        # label = tk.Label(self, image=photo, bg='black')
        # label.image = photo  # keep a reference!
        # label.grid(row=1, columnspan = 10, sticky='e')


        scales, i = pleas_rate_art(self, ['a', 'b', 'a_and_b'])

        next_button(self, scales, controller, artTwo, i)


class artTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        image = Image.open(path + 'painting_info1' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=0, columnspan=10, sticky='e')

        agree_with(self, controller, artThree, n = 2)


class artThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        image = Image.open(path + 'art_rate' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=1, columnspan = 10, sticky='e')

        scales, i = pleas_rate_art(self, ['a', 'd', 'a_and_d'])

        next_button(self, scales, controller, artFour, i)


class artFour(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        image = Image.open(path + 'painting_info2' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=0, columnspan=10, sticky='e')

        agree_with(self, controller, artFive, n = 2)


class artFive(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        image = Image.open(path + 'painting_rank_all' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=1, columnspan=4, sticky='e')

        rankings = {}
        w,h = 401, 46
        for i, photo in enumerate(['a','b', 'c', 'd']):
            rankings[photo] = tk.Entry(self,  width = 5)
            rankings[photo].grid(row=i + 2, column=2, pady=0, sticky='n')

            image = Image.open(path + 'art_' + photo + '.png')
            # w, h = image.size
            image = image.resize((w, h), Image.ANTIALIAS)  # The (250, 250) is (height, width)
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(self, image=photo, bg='black')
            label.image = photo  # keep a reference!
            label.grid(row=i + 2, column=3, sticky='e', pady=0)

        for i, photo in enumerate(['a_and_b', 'a_and_d', 'b_and_c', 'b_and_d', 'c_and_d']):
            rankings[photo] = tk.Entry(self, width=5)
            rankings[photo].grid(row=i + 2, column=0, pady=0, sticky='n')

            image = Image.open(path + 'art_' + photo + '.png')
            image = image.resize((w, h), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(self, image=photo, bg='black')
            label.image = photo  # keep a reference!
            label.grid(row=i + 2, column=1, sticky='s', padx=0)

        # button1 = tk.Button(self, text="<--", width=20,
        #                     command=lambda: controller.show_frame(PageSix))
        # button1.grid(row=i + 2, column=1, columnspan=1)

        next_button(self, rankings, controller, artSix, i+2)


class artSix(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        image = Image.open(path + 'robots_finshed_talking' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=0, columnspan = 2, sticky='e')

        image = Image.open(path + 'which_painting_was_sold' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=1, columnspan=2, sticky='e')

        rankings = {}
        w,h = 401, 46
        for i, p in enumerate(['a','b', 'c', 'd']):
            image = Image.open(path + 'art_' + p + '.png')
            image = image.resize((w, h), Image.ANTIALIAS)  # The (250, 250) is (height, width)
            photo = ImageTk.PhotoImage(image)
            rankings[p] = tk.Button(self, image=photo, bg='black',command=lambda p=p: transition([p], controller, artSeven))
            rankings[p].image = photo  # keep a reference!
            rankings[p].grid(row=i + 2, column=1, sticky='s', padx=0, pady = 5)

        for i, p in enumerate(['a_and_b', 'a_and_d', 'b_and_c', 'b_and_d', 'c_and_d']):
            image = Image.open(path + 'art_' + p + '.png')
            image = image.resize((w, h), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            rankings[p] = tk.Button(self, image=photo, bg='black',command=lambda p=p: transition([p], controller, artSeven))
            rankings[p].image = photo  # keep a reference!
            rankings[p].grid(row=i + 2, column=0, sticky='s', padx=0, pady = 5)


class artSeven(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        n = -1
        image = Image.open(path + 'which_robot_hire' + '.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=n + 1, columnspan=10, sticky='e', pady=10)

        image = Image.open(path + 'red' + '.png')
        photo = ImageTk.PhotoImage(image)
        clr = '#%02x%02x%02x' % (255, 80, 80)
        red_button = tk.Button(self, image=photo, bg=clr,
                               command=lambda: transition(['red'], controller, EndPage, p = 'art'))
        red_button.image = photo
        red_button.grid(row=n + 2, column=1, pady=30)

        image = Image.open(path + 'blue' + '.png')
        photo = ImageTk.PhotoImage(image)
        clr = '#%02x%02x%02x' % (47, 85, 151)
        blue_button = tk.Button(self, image=photo, bg=clr,
                                command=lambda: transition(['blue'], controller, EndPage, p = 'art'))
        blue_button.image = photo
        blue_button.grid(row=n + 2, column=8, pady=30)


class OpeningPage(tk.Frame):
    global first_story, Gender
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        image = Image.open(path + 'opening_%s.png' % Gender)
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=0, sticky='e', pady=10)

        clr = '#%02x%02x%02x' % (146, 208, 80)
        image = Image.open(path + 'begin_button' + '.png')
        photo = ImageTk.PhotoImage(image)

        b = tk.Button(self, image=photo, bg=clr, command=lambda: transition(None, controller, [PageOne, artOne], parent.master.gender))

        b.grid(row=1, pady=10)
        b.image = photo


class case2(tk.Frame):
    global first_story
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        image = Image.open(path + 'case2.png')
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(self, image=photo, bg='black')
        label.image = photo  # keep a reference!
        label.grid(row=0, sticky='e', pady=10)

        clr = '#%02x%02x%02x' % (146, 208, 80)
        image = Image.open(path + 'begin_button' + '.png')
        photo = ImageTk.PhotoImage(image)

        b = tk.Button(self, image=photo, bg=clr, command=lambda: transition(None, controller, [PageOne, artOne], parent.master.gender, p = 'case2'))

        b.grid(row=1, pady=10)
        b.image = photo


# app = SeaofBTCapp(test = True) # uncomment for real run
app = SeaofBTCapp()
app.mainloop()

# relative positioning --> independent of the screen size --> better for full screen
# https://www.python-course.eu/tkinter_layout_management.phpv --> see *.place()
# todo: torr remember you need it to work not to be the prettiest

