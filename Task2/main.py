import tkinter as tk
import warnings
from tkinter import *
from tkinter import messagebox
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from GUI import *
from algoritms import *
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


def fetch():
    excel_file = "Dry_Bean_Dataset.xlsx"
    df = pd.read_excel(excel_file)

    # data preprocessing
    count_null_in_Area = df['Area'].isnull().sum()
    count_null_in_Perimeter = df['Perimeter'].isnull().sum()
    count_null_in_MajorAL = df['MajorAxisLength'].isnull().sum()
    count_null_in_MinorAL = df['MinorAxisLength'].isnull().sum()  # 1 columns is null
    count_null_in_roundness = df['roundnes'].isnull().sum()

    mean_MinorAxisLength = df['MinorAxisLength'].head(50).mean()
    df['MinorAxisLength'].fillna(mean_MinorAxisLength, inplace=True)
    print(f"Is there any duplicates:{df.duplicated().any()}")  # no duplicates

    # data splitting
    dfClassOne = df[(df['Class'] == "BOMBAY")]
    dfClassTwo = df[(df['Class'] == "CALI")]
    dfClassThree = df[(df['Class'] == "SIRA")]


    features = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']
    X1_train, X1_test, y1_train, y1_test = train_test_split(dfClassOne[features], dfClassOne['Class'],
                                                            test_size=0.4, random_state=42, shuffle=True)
    X2_train, X2_test, y2_train, y2_test = train_test_split(dfClassTwo[features], dfClassTwo['Class'],
                                                            test_size=0.4, random_state=42, shuffle=True)
    X3_train, X3_test, y3_train, y3_test = train_test_split(dfClassThree[features], dfClassThree['Class'],
                                                            test_size=0.4, random_state=42, shuffle=True)

    # rows gathering
    X_train = np.concatenate((X1_train, X2_train, X3_train), axis=0)
    X_test = np.concatenate((X1_test, X2_test, X3_test), axis=0)
    yTrain = np.concatenate((y1_train, y2_train, y3_train), axis=0)
    yTest = np.concatenate((y1_test, y2_test, y3_test), axis=0)

    # target Encoding
    encoder = OneHotEncoder()
    yTrain = encoder.fit_transform(yTrain.reshape(-1, 1)).toarray()
    yTest = encoder.transform(yTest.reshape(-1, 1)).toarray()

    # Normalization
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ------------Reading variables
    NeuronsPerLayer = []
    try:
        HiddenLayers = num_hidden_layers.get()
    except Exception as f:
        messagebox.showerror("Error", f"An error occurred in Entry number of hidden layers: {str(f)}")
        return
    try:
        inputString = num_neurons.get()
        values = inputString.split(",")  # Split the input string using a comma delimiter
        values = [int(value.strip()) for value in values]  # Convert the values to integers
        NeuronsPerLayer.extend(values)  # Extend the integer_values list with the extracted values
        if len(NeuronsPerLayer) != HiddenLayers:
            raise ValueError("The number of neurons per layer is not correct")
    except Exception as g:
        messagebox.showerror("Error", f"An error occurred in Entry number of neurons in each hidden layer: {str(g)}")
        return
    try:
        lr = float(learning_rate.get())
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred in Entry learning rate: {str(e)}")
        return
    try:
        epochs_Number = epochsNumber.get()
        if epochs_Number <= 0:
            raise ValueError("The epochs number must be positive value")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred in Entry epochs number: {str(e)}")
        return
    try:
        flag = flag1.get()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred in Entry with bias: {str(e)}")
        return
    try:
        actFn = activationFn.get()  # 0->sigmoid,1->Hyperbolic Tangent sigmoid
        if actFn != '0' and actFn != '1':
            raise ValueError("Please choose an activation function")


    except Exception as e:
        messagebox.showerror("Error", f"An error occurred in Entry activation function: {str(e)}")
        return

    input_w, hidden_w, output_w = Back_propagation(X_train, yTrain, HiddenLayers, NeuronsPerLayer, lr, epochs_Number, flag, actFn)
    confusion_matrix1 =test(X_train,yTrain,input_w, hidden_w, output_w, HiddenLayers, NeuronsPerLayer, flag, actFn)
    confusion_matrix2 = test(X_test, yTest, input_w, hidden_w, output_w, HiddenLayers, NeuronsPerLayer, flag, actFn)

    # confusion_matrix = np.array([[tp[0], fp[0], fn[0]],
    #                              [tp[1], fp[1], fn[1]],
    #                              [tp[2], fp[2], fn[2]]])


    ConfusionMatrix = pd.DataFrame(data=confusion_matrix1)
    plt.subplots(figsize=(12, 8))
    sns.heatmap(ConfusionMatrix, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), fmt='d',
                xticklabels=['BOMBAY', 'CALI', 'SIRA'],
                yticklabels=['BOMBAY', 'CALI', 'SIRA'])
    plt.title("Confusion Matrix for training")
    plt.show()

    ConfusionMatrix = pd.DataFrame(data=confusion_matrix2)
    plt.subplots(figsize=(12, 8))
    sns.heatmap(ConfusionMatrix, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), fmt='d',
                xticklabels=['BOMBAY', 'CALI', 'SIRA'],
                yticklabels=['BOMBAY', 'CALI', 'SIRA'])
    plt.title("Confusion Matrix for testing")
    plt.show()


    testingAccuracy = np.trace(confusion_matrix2) / len(yTest)
    trainingAccuracy = np.trace(confusion_matrix1) / len(yTrain)
    messagebox.showinfo('Training Accuracy: ', trainingAccuracy * 100)
    messagebox.showinfo('Testing Accuracy: ', testingAccuracy * 100)
    # ----------------------------------------------------------GUI-----------------------------------------------------

    def newWindow():
        gui.destroy()
        gui2 = tk.Tk()
        gui2.geometry("800x500")
        gui2.resizable(True, True)

        # variables
        f1 = StringVar()
        f2 = StringVar()
        f3 = StringVar()
        f4 = StringVar()
        f5 = StringVar()

        Label(gui2, text="Enter value of first :").grid(row=0, column=0)
        Entry(gui2, textvariable=f1).grid(row=0, column=2)
        Label(gui2, text="Enter value of second :").grid(rows=1, column=0)
        Entry(gui2, textvariable=f2).grid(row=1, column=2)
        Label(gui2, text="Enter value of third :").grid(rows=2, column=0)
        Entry(gui2, textvariable=f3).grid(row=2, column=2)
        Label(gui2, text="Enter value of fourth :").grid(rows=3, column=0)
        Entry(gui2, textvariable=f4).grid(row=6, column=2)
        Label(gui2, text="Enter value of fifth :").grid(rows=4, column=0)
        Entry(gui2, textvariable=f5).grid(row=8, column=2)

        def predict():
            try:
                x1 = float(f1.get())
                x2 = float(f2.get())
                x3 = float(f3.get())
                x4 = float(f4.get())
                x5 = float(f5.get())
                data = scaler.transform([[x1, x2, x3, x4, x5]])
                data=np.array(data)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
                return

            if flag:
                data = np.append(data,1)
                print(data)

            out1,out2,outputlayeroutputs = forward_step(input_w, hidden_w, output_w, data, HiddenLayers, NeuronsPerLayer, flag, actFn)
            max_index = outputlayeroutputs.index(max(outputlayeroutputs))
            prediction = [0] * len(outputlayeroutputs)
            prediction[max_index] = 1
            prediction = np.array(prediction)
            if np.all(prediction == [1,0,0]):
                messagebox.showinfo('It belongs to:', "BOMBAY")
            elif np.all(prediction==[0,1,0]):
                messagebox.showinfo('It belongs to:', 'CALI')
            elif np.all(prediction==[0,0,1]):
                messagebox.showinfo('It belongs to:', 'SIRA')
        actionButton = Button(gui2, text='Predict', command=predict)
        actionButton.place(relx=0.3, rely=0.7, relwidth=0.4, relheight=0.1)
        gui2.mainloop()

    actionButton = Button(gui, text='try Sample', command=newWindow)
    actionButton.place(relx=0.5, rely=0.7, relwidth=0.4, relheight=0.1)


actionButton = Button(gui, text='Run Algorithm', command=fetch)
actionButton.place(relx=0.1, rely=0.7, relwidth=0.4, relheight=0.1)
gui.mainloop()