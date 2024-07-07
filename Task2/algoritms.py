import numpy as np
from sklearn.metrics import confusion_matrix

def init_weight(HiddenLayers, NeuronsPerLayer, flag):
    input_w = np.zeros((5+(1*flag), NeuronsPerLayer[0]))
    for i in range(5 + (1*flag)):
        for j in range(NeuronsPerLayer[0]):
            input_w[i, j] = np.random.rand()


    hidden_w = []

    for i in range((HiddenLayers-1)):
        EachLayerW=[]
        for k in range(NeuronsPerLayer[i+1]):
            EachNeuronsW = []
            for j in range(NeuronsPerLayer[i]+(1*flag)):
                tempW = np.random.rand()
                EachNeuronsW.append(tempW)
            EachLayerW.append(EachNeuronsW)
        hidden_w.append(EachLayerW)



    output_w = np.zeros((NeuronsPerLayer[len(NeuronsPerLayer)-1] + (1*flag), 3))
    for i in range(NeuronsPerLayer[len(NeuronsPerLayer)-1] + (1*flag)):
        for j in range(3):
            output_w[i, j] = np.random.rand()


    return input_w, hidden_w, output_w


def sigmoidAF(x):
    return 1 / (1 + np.exp(-x))


def TanhAF(x):
    return np.tanh(x)


def forward_step(input_w, hidden_w, output_w, X_train, HiddenLayers, NeuronsPerLayer, flag, actFn):
    # ---input->hidden-----------
    output1h_array = []  # output values for first hidden layer
    for j in range(NeuronsPerLayer[0]):
        firstLayerOutput = np.dot(input_w[:, j], X_train)
        if actFn == "0":  # sigmoid
            output1h = sigmoidAF(firstLayerOutput)
            output1h_array.append(output1h)
        else:
            output1h = TanhAF(firstLayerOutput)
            output1h_array.append(output1h)

    # --------hidden->hidden----------
    HiddenLayersOutputs = []   # 2d array where [hidden layer number, node number]
    HiddenLayersOutputs.append(output1h_array)

    inputArray = output1h_array.copy()
    if flag:
        inputArray.append(1)

    for j in range(HiddenLayers - 1):
        outputHH_array = []
        for k in range(NeuronsPerLayer[j+1]):
            init_out = np.dot(hidden_w[j][k], inputArray)
            if actFn == "0":
                outputHH = sigmoidAF(init_out)
                outputHH_array.append(outputHH)
            else:
                outputHH = TanhAF(init_out)
                outputHH_array.append(outputHH)
        HiddenLayersOutputs.append(outputHH_array)
        inputArray = outputHH_array.copy()
        if flag:
            inputArray.append(1)

    # --------------hidden->output--------------------

    outputLayeroutputs = []  # 1d array represent outputs of each node in last layer
    if HiddenLayers == 1:
        outputHH_array = output1h_array.copy()
    if flag:
        outputHH_array.append(1)

    for j in range(3):
        init_out2 = np.dot(output_w[:, j], outputHH_array)
        if actFn == "0":
            outputHO = sigmoidAF(init_out2)
            outputLayeroutputs.append(outputHO)
        else:
            outputHO = TanhAF(init_out2)
            outputLayeroutputs.append(outputHO)

    return output1h_array, HiddenLayersOutputs, outputLayeroutputs

def Back_propagation(X_train,ytrain, HiddenLayers, NeuronsPerLayer, lr, epochs_Number, flag, actFn):

    input_w, hidden_w, output_w = init_weight(HiddenLayers, NeuronsPerLayer, flag)
    # -------------------------------------forward step--------------------------------------
    if flag:
        ones_column = np.ones((X_train.shape[0], 1))
        X_train = np.concatenate((X_train, ones_column), axis=1)

    for i in range(epochs_Number):
        for j in range(X_train.shape[0]):

            output1h_array, HiddenLayersOutputs, outputLayeroutputs = forward_step(input_w, hidden_w, output_w, X_train[j], HiddenLayers, NeuronsPerLayer, flag, actFn)

            outputLayerError = []  # 1D array (1 x 3)
            onearray = np.array([1, 1, 1])
            outputLayeroutputs = np.array(outputLayeroutputs)
            y_train_j = np.array(ytrain[j])

            # calculate error at each neuron in outputLayer

            if actFn == "0":
                outputLayerError = (y_train_j - outputLayeroutputs) * outputLayeroutputs * (onearray- outputLayeroutputs)
            else:
                outputLayerError = (y_train_j - outputLayeroutputs) * (onearray - (outputLayeroutputs)**2)  # s1 s2 s3

            # HiddenLayer <-- OutputLayer
            # calculate error at each neuron in last hidden layer
            LastLayerError = [] # s3 s4 s5 . . . s[NeuronsPerLayer[len(NeuronsPerLayer) - 1]
            allhiddenLayersError = []
            for f in range(NeuronsPerLayer[len(NeuronsPerLayer) - 1]):
                error = np.dot(outputLayerError, output_w[f, :])
                if actFn == "0":
                    newSegma = HiddenLayersOutputs[HiddenLayers - 1][f] * (1 - HiddenLayersOutputs[HiddenLayers - 1][f]) *error
                else:
                    newSegma = (1 - (HiddenLayersOutputs[HiddenLayers - 1][f])**2) * error
                LastLayerError.append(newSegma)

            allhiddenLayersError.append(LastLayerError)

            rowNo = 0
            for layer in range(HiddenLayers-2, -1, -1):
                sigma = np.zeros(NeuronsPerLayer[layer]+(1*flag))
                for ne in range(0, NeuronsPerLayer[layer + 1]):
                    for w in range(len(hidden_w[layer][ne])):
                        sigma[w] += hidden_w[layer][ne][w] * allhiddenLayersError[rowNo][ne]

                for s in range(0, NeuronsPerLayer[layer]):
                    if actFn == "0":
                        sigma[s] = sigma[s] * HiddenLayersOutputs[layer][s] * (1 - HiddenLayersOutputs[layer][s])
                    else:
                        sigma[s] = sigma[s] * (1 - (HiddenLayersOutputs[layer][s])**2)

                allhiddenLayersError.append(sigma)
                rowNo +=1

            # Update output_w

            for n in range(NeuronsPerLayer[len(NeuronsPerLayer) - 1] + (1 * flag)):
                for o in range(3):
                    if flag and n == NeuronsPerLayer[len(NeuronsPerLayer) - 1]:
                        output_w[n][o] = output_w[n][o] + lr * outputLayerError[o]
                    else:
                        output_w[n][o] = output_w[n][o] + lr * outputLayerError[o] * HiddenLayersOutputs[HiddenLayers -1][n]


            # update input_w
            lastArrayIndex = len(allhiddenLayersError) - 1
            for a in range(5 + (1 * flag)):
                for b in range(NeuronsPerLayer[0]):
                    input_w[a][b] = input_w[a][b] + lr * allhiddenLayersError[lastArrayIndex][b] * X_train[j][a]

            # update hidden_w
            CurrentArrayIndex = len(allhiddenLayersError) - 2
            for hl in range(HiddenLayers -1):

                for n1 in range(NeuronsPerLayer[hl+1]):

                    for n2 in range(NeuronsPerLayer[hl] + (1*flag)):
                        if flag and n2 == NeuronsPerLayer[hl]:
                            hidden_w[hl][n1][n2] = hidden_w[hl][n1][n2] + lr * allhiddenLayersError[CurrentArrayIndex][n1]
                        else:
                            hidden_w[hl][n1][n2] = hidden_w[hl][n1][n2] + lr * allhiddenLayersError[CurrentArrayIndex][n1] * HiddenLayersOutputs[hl][n2]
                CurrentArrayIndex -= 1

    return input_w, hidden_w, output_w







def test(X_test, ytest, input_w, hidden_w, output_w, HiddenLayers, NeuronsPerLayer, flag,actFn):
    mapped_y_predicted = [] #2d array (60, 3)
    y_predicted = [] #1d array (60, 1)
    old = []
    if flag:
        ones_column = np.ones((X_test.shape[0], 1))
        X_test = np.concatenate((X_test, ones_column), axis=1)
    for i in range(X_test.shape[0]):
        output1h_array, HiddenLayersOutputs, outputLayeroutputs = forward_step(input_w, hidden_w, output_w, X_test[i],HiddenLayers, NeuronsPerLayer, flag,actFn)
        old.append(outputLayeroutputs)
        max_index = outputLayeroutputs.index(max(outputLayeroutputs))
        new_array = [0] * len(outputLayeroutputs)
        new_array[max_index] = 1
        y_predicted.append(new_array)


    ytest = np.array(ytest)
    print(f"ytest:{ytest}")
    y_predicted = np.array(y_predicted)
    print(f"y predicted:{y_predicted}")
    print("Unique classes in ytest:", np.unique(ytest))
    print("Unique classes in y_predicted:", np.unique(y_predicted))

    # # Compare element-wise and count the number of correct predictions
    # correct_predictions = np.sum(np.all(ytest == y_predicted, axis=1))
    #
    # # Calculate accuracy
    # accuracy = correct_predictions / len(ytest)*100

    confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for test in range(len(y_predicted)):
        if np.all(ytest[test] == [1,0,0]):
            actual_class = 0
        elif np.all(ytest[test] == [0,1,0]):
            actual_class = 1
        else:
            actual_class = 2

        if np.all(y_predicted[test] == [1, 0, 0]):
            predicted_class = 0
        elif np.all(y_predicted[test] == [0, 1, 0]):
            predicted_class = 1
        else:
            predicted_class = 2

        confusion_matrix[actual_class][predicted_class] += 1

    # # Calculate metrics
    # TP = confusion_matrix[1][1]  # True positive for class B
    # TN = confusion_matrix[0][0] + confusion_matrix[0][2] + confusion_matrix[2][0] + confusion_matrix[2][2]  # True negative for classes A and C
    # FP = confusion_matrix[0][1] + confusion_matrix[0][2] + confusion_matrix[1][0] + confusion_matrix[2][0] + confusion_matrix[1][2] + confusion_matrix[2][1]  # False positive for classes A, B, and C
    # FN = confusion_matrix[1][0] + confusion_matrix[1][2] + confusion_matrix[2][1] + confusion_matrix[2][0]  # False negative for classes A and C
    #
    # accuracy = 0
    # precision = 0
    # recall = 0
    # f1_score = 0
    #
    # if (TP + TN + FP + FN) != 0:
    #     accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100
    # if (TP + FP) != 0:
    #     precision = TP / (TP + FP)
    # if (TP + FN) != 0:
    #     recall = TP / (TP + FN)
    # if (precision + recall) != 0:
    #     f1_score = 2 * (precision * recall) / (precision + recall)
    #
    # print('accuracy = ', accuracy , '%')
    # print('precision = ', precision)
    # print('recall = ', recall)
    # print('f1_score = ', f1_score)



    return confusion_matrix


