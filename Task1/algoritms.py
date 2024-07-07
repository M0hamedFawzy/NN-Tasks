import numpy as np


def signum(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def perceptron_lr(W, x1, x2, lr, epochs, target, b, flag):
    for i in range(epochs):
        for j in range(len(x1)):
            z = W[0] * x1[j] + W[1] * x2[j] + (b * flag)
            y = signum(z)
            if y != target[j]:
                loss = (target[j] - y)
                W[0] = W[0] + lr * loss * x1[j]
                W[1] = W[1] + lr * loss * x2[j]
                b = (b + lr * loss) * flag
    return W, b


def adaline(W, x1, x2, lr, epochs, target, MSEthreshold, b, flag):
    for i in range(epochs):
        for j in range(len(x1)):
            y = W[0] * x1[j] + W[1] * x2[j] + (b * flag)
            if y != target[j]:
                loss = (target[j] - y)
                W[0] = W[0] + lr * loss * x1[j]
                W[1] = W[1] + lr * loss * x2[j]
                b = (b + lr * loss) * flag

        # MSE
        y = np.dot(W, [x1, x2]) + b
        error = 0.5 * np.mean((target - y) ** 2)

        if error < MSEthreshold:
            return W

    return W, b


def test(x1, x2, y, outputWeights, b):
    tp, tn, fp, fn = 0, 0, 0, 0
    accurate = 0
    for i in range(len(x1)):
        y_pred = outputWeights[0] * x1[i] + outputWeights[1] * x2[i] + b
        y_pred = signum(y_pred)

        if y[i] == 1:
            if y_pred == 1:
                tp += 1
                accurate += 1
            else:
                fn += 1
        else:
            if y_pred == 1:
                fp += 1

            else:
                tn += 1
                accurate += 1

    return tp, fp, tn, fn, accurate
