import importlib as im
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
# ---------------------------------GUI----------------------------------

gui = tk.Tk()
gui.title("Run Algorithm")
# Defining variables --->will be taken from user
featureOne = StringVar()
featureTwo = StringVar()
classOne = StringVar()
classTwo = StringVar()
learning_rate = StringVar()
epochsNumber = IntVar()
mse_threshold = StringVar()
flag1 = IntVar()  # flag
alg = StringVar()
gui.geometry("800x500")
gui.resizable(False, False)

variables=('Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes')
ff = Label(gui, text="First Feature").grid(row=0, column=0)
ff1= Combobox(gui, width = 15,state='readonly',
          textvariable = featureOne,values=variables).grid(row=0, column=2)
Sf = Label(gui, text="Second Feature").grid(row=1, column=0)
sf = Combobox(gui, width = 15,state='readonly',
          textvariable = featureTwo,values=variables).grid(row=1, column=2)

fc = Label(gui, text="First Class").grid(row=2, column=0)
fc1 = Radiobutton(gui, text="Bombay", variable=classOne, value="BOMBAY").grid(row=2, column=1)
fc2 = Radiobutton(gui, text="Cali", variable=classOne, value="CALI").grid(row=2, column=2)
fc3 = Radiobutton(gui, text="Sira", variable=classOne, value="SIRA").grid(row=2, column=3)

sc = Label(gui, text="Second Class").grid(row=3, column=0)
sc1 = Radiobutton(gui, text="Bombay", variable=classTwo, value="BOMBAY").grid(row=3, column=1)
sc2 = Radiobutton(gui, text="Cali", variable=classTwo, value="CALI").grid(row=3, column=2)
sc3 = Radiobutton(gui, text="Sira", variable=classTwo, value="SIRA").grid(row=3, column=3)

li1 = Label(gui, text="learning Rate").grid(row=4, column=0)
li = Entry(gui, textvariable=learning_rate).grid(row=4, column=2)

ep = Label(gui, text="Epochs Number").grid(row=5, column=0)
ep1 = Entry(gui, textvariable=epochsNumber).grid(row=5, column=2)

ms = Label(gui, text="MSE Threshold").grid(row=6, column=0)
ms1 = Entry(gui, textvariable=mse_threshold).grid(row=6, column=2)

flG = Label(gui, text="With Bias?").grid(row=7, column=0)
flG1 = tk.Checkbutton(gui, variable=flag1, onvalue=1, offvalue=0).grid(row=7, column=1)

Label(gui, text="Algorithm").grid(row=8, column=0)
alg1 = Radiobutton(gui, text="Perceptron", variable=alg, value="Perceptron").grid(row=8, column=2)
alg2 = Radiobutton(gui, text="Adaline", variable=alg, value="Adaline").grid(row=8, column=3)
