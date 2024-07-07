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

    class_One = classOne.get()
    class_Two = classTwo.get()
    Feature_One = featureOne.get()
    Feature_Two = featureTwo.get()

    df = df[(df['Class'] == class_One) | (df['Class'] == class_Two)]
    df = df[[Feature_One, Feature_Two, 'Class']]

    # target classes encoding
    # encoder = LabelEncoder()
    # df['Class'] = encoder.fit_transform(df['Class'])
    # print(df)

    dfClassOne = df[((df['Class'] == class_One))]
    dfClassOne.loc[:, 'Class'] = -1  # law 3mlna da alencoder malosh lazma
    dfClassTwo = df[((df['Class'] == class_Two))]
    dfClassTwo.loc[:, 'Class'] = 1

    # data splitting
    X1_train, X1_test, y1_train, y1_test = train_test_split(dfClassOne[[Feature_One, Feature_Two]], dfClassOne['Class'],
                                                            test_size=0.4, random_state=42, shuffle=True)
    X2_train, X2_test, y2_train, y2_test = train_test_split(dfClassTwo[[Feature_One, Feature_Two]], dfClassTwo['Class'],
                                                            test_size=0.4, random_state=42, shuffle=True)

    # rows gathering
    xTrain = np.concatenate((X1_train, X2_train), axis=0)
    yTrain = np.concatenate((y1_train, y2_train), axis=0)
    xTest = np.concatenate((X1_test, X2_test), axis=0)
    yTest = np.concatenate((y1_test, y2_test), axis=0)

    # Normalization
    scaler = MinMaxScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)

    # columns splitting
    featureOne_train = xTrain[:, 0]
    featureTwo_train = xTrain[:, 1]
    featureOne_test = xTest[:, 0]
    featureTwo_test = xTest[:, 1]

    # generate random values for weights [W1 , W2] and bias
    w1_1 = np.random.rand()
    w2_1 = np.random.rand()
    W_vec = np.array([w1_1, w2_1])
    bias = 0.05

    lr = float(learning_rate.get())
    epochs_Number = epochsNumber.get()
    mse_threshold.set(0)
    MSE_Threshold = float(mse_threshold.get())
    flag = flag1.get()
    algorithm = alg.get()

    if algorithm == 'Adaline':
        outputWeights, bias1 = adaline(W_vec, featureOne_train, featureTwo_train, lr, epochs_Number, yTrain,
                                       MSE_Threshold, bias, flag)
    else:
        outputWeights, bias1 = perceptron_lr(W_vec, featureOne_train, featureTwo_train, lr, epochs_Number, yTrain, bias,
                                             flag)

    tp1, fp1, tn1, fn1, accurate_train = test(featureOne_train,featureTwo_train,yTrain,outputWeights,bias1)
    messagebox.showinfo('Train Accuracy', (accurate_train / len(featureOne_train)) * 100)

    print(f'{algorithm} weights --> ', outputWeights)
    print(f'{algorithm} bias --> ', bias1)

    feat_One_data = featureOne_train[:30], featureTwo_train[:30]
    feat_Two_data = featureOne_train[30:], featureTwo_train[30:]

    # plotting

    W1 = outputWeights[0]
    W2 = outputWeights[1]

    X1 = np.linspace(0, 1, 100)
    X2 = -(W1 / W2) * X1 - (bias1 / W2)

    plt.scatter(feat_One_data[0], feat_One_data[1], label=class_One, color='orange', marker='o',
                s=50)  # class 1 points plot
    plt.scatter(feat_Two_data[0], feat_Two_data[1], label=class_Two, color='green', marker='x',
                s=50)  # class 2 points plot
    plt.plot(X1, X2, label=f'{algorithm} Separation Boundary', color='red')  # decision boundry plot

    plt.xlabel(f'{Feature_One}')
    plt.ylabel(f'{Feature_Two}')
    plt.legend()
    plt.grid(True)
    plt.show()

    tp, fp, tn, fn, accurate = test(featureOne_test, featureTwo_test, yTest, outputWeights, bias1)

    ConfusionMatrix = pd.DataFrame(data=np.array([[tp, fp], [fn, tn]]))
    plt.subplots(figsize=(8, 4))
    sns.heatmap(ConfusionMatrix, annot=True, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.title("Confusion Matrix")
    plt.show()
    messagebox.showinfo('Test Accuracy', (accurate / len(featureOne_test)) * 100)


    def newWindow():
        gui.destroy()
        gui2 = tk.Tk()
        gui2.geometry("800x500")
        gui2.resizable(True, True)

        # variables
        f1 = StringVar()
        f2 = StringVar()

        l = Label(gui2, text="Enter value of first feature:").grid(row=0, column=2)
        s = Entry(gui2, textvariable=f1).grid(row=0, column=5)
        l2 = Label(gui2, text="Enter value of second feature:").grid(rows=2, column=2)
        s2 = Entry(gui2, textvariable=f2).grid(row=2, column=5)

        def predict():
            x1 = float(f1.get())
            x2 = float(f2.get())
            data = scaler.transform([[x1, x2]])

            pred = (outputWeights[0] * data[0, 0]) + (outputWeights[1] * data[0, 1]) + bias1
            pred = signum(pred)

            if pred == -1:
                messagebox.showinfo('It belongs to:', class_One)
            else:
                messagebox.showinfo('It belongs to:', class_Two)

        actionButton = Button(gui2, text='Predict', command=predict)
        actionButton.place(relx=0.3, rely=0.7, relwidth=0.4, relheight=0.1)
        gui2.mainloop()

    actionButton = Button(gui, text='try Sample', command=newWindow)
    actionButton.place(relx=0.5, rely=0.7, relwidth=0.4, relheight=0.1)


actionButton = Button(gui, text='Run Algorithm', command=fetch)
actionButton.place(relx=0.1, rely=0.7, relwidth=0.4, relheight=0.1)
gui.mainloop()
