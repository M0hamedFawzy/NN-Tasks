import importlib as im
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
# ---------------------------------GUI----------------------------------

gui = tk.Tk()
gui.title("Run Algorithm")
# Defining variables --->will be taken from user
num_hidden_layers = IntVar()
num_neurons = StringVar()
learning_rate = StringVar()  # it is string to be able to turn it into float
epochsNumber = IntVar()
flag1 = IntVar()  # bias flag
activationFn = StringVar()
gui.geometry("800x500")
gui.resizable(False, False)

Label(gui, text="Enter number of hidden layers:").grid(row=0,column=0)
Entry(gui, textvariable=num_hidden_layers).grid(row=0, column=2)

Label(gui, text="Enter number of neurons in each hidden layer(separated with comma):").grid(row=1, column=0)
Entry(gui, textvariable=num_neurons).grid(row=1, column=2)

Label(gui, text="learning Rate").grid(row=4, column=0)
Entry(gui, textvariable=learning_rate).grid(row=4, column=2)

Label(gui, text="Epochs Number").grid(row=5, column=0)
Entry(gui, textvariable=epochsNumber).grid(row=5, column=2)

Label(gui, text="With Bias?").grid(row=7, column=0)
tk.Checkbutton(gui, variable=flag1, onvalue=1, offvalue=0).grid(row=7, column=1)

Label(gui, text="Activation function").grid(row=8, column=0)
Radiobutton(gui, text="Sigmoid", variable=activationFn, value="0").grid(row=8, column=2)
Radiobutton(gui, text="Hyperbolic Tangent sigmoid ", variable=activationFn, value="1").grid(row=8, column=3)
