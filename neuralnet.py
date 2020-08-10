import numpy as np
import sys


class Object:
    def __init__(self, x, a, z, b, y_hat, J):
        self.a = a
        self.x = x
        self.z = z
        self.b = b
        self.y_hat = y_hat
        self.J = J


def LinearForward(x, alpha):
    return np.dot(alpha, np.array(x).reshape(len(x), 1))


def LinearBackward(z, b, g_b, beta):
    g_beta = np.dot(g_b, z.T)
    g_z = np.dot(beta[:, 1:].T, g_b)
    return g_beta, g_z


def SigmoidForward(x):
    return 1 / (1 + np.exp(-x))


def SigmoidBackward(a, z, g_z):
    return np.multiply(np.multiply(g_z, z[1:, :]), (1 - z[1:, :]))


def SoftmaxForward(x):
    denominator = np.sum(np.exp(x))
    y_hat = np.exp(x) / denominator
    return y_hat


def SoftmaxBackward(b, y_hat, g_y_hat):
    print("y_hat:")
    print(np.diagflat(y_hat.reshape(-1, 1)))
    print("dot product:")
    print(np.dot(y_hat, y_hat.T))
    softmaxderivative = np.diagflat(y_hat.reshape(-1, 1)) - np.dot(y_hat, y_hat.T)
    return np.dot(g_y_hat.T, softmaxderivative).T


def CrossEntropyForward(y, y_hat):
    return -1 * np.dot(y, np.log(y_hat))


def CrossEntropyBackward(y, y_hat, J, g_j):
    vec_y = np.array(y).reshape(len(y), 1)
    return -1 * (vec_y / y_hat)


def sigmoid_derivative(z):
    return z * (1 - z)


def NNforward(x, y, alpha, beta):
    a = LinearForward(np.array(x).T, alpha)
    print("a:")
    print(a)
    z = SigmoidForward(a)
    print("z:")
    print(z)
    z_bias = np.array(np.append(1.0, z)).reshape(len(z) + 1, 1)
    b = LinearForward(z_bias, beta)
    print("b:")
    print(b)
    y_hat = SoftmaxForward(b)
    print("y_hat:")
    print(y_hat)
    J = CrossEntropyForward(y, y_hat)
    print("J:")
    print(J)
    o = Object(x, a, z_bias, b, y_hat, J)
    return o


def NNbackward(x, y, alpha, beta, o):
    x = o.x
    a = o.a
    z = o.z
    b = o.b
    y_hat = o.y_hat
    J = o.J
    g_j = 1
    g_y_hat = CrossEntropyBackward(y, y_hat, J, g_j)
    print("g_y_hat:")
    print(g_y_hat)
    g_b = SoftmaxBackward(b, y_hat, g_y_hat)
    print("g_b:")
    print(g_b)
    g_beta, g_z = LinearBackward(z, b, g_b, beta)
    print("g_beta:")
    print(g_beta)
    g_a = SigmoidBackward(a, z, g_z)
    print("g_a:")
    print(g_a)

    print("x:")
    print(x)
    print("x.T")
    print(np.array([x]))

    g_alpha, g_x = LinearBackward(np.array([x]).T, a, g_a, alpha)
    print("g_alpha:")
    print(g_alpha.T)
    return g_alpha, g_beta


def stocastic_gradient_descent(train_data, train_label, hidden_units, test_data, test_label, epochs, lr, flag_init, metrics_out):
    # initialize parameters
    if flag_init == '2':
        alpha = zeros_weight_init(hidden_units, len(train_data[0]))
        beta = zeros_weight_init(len(train_label[0]), hidden_units + 1)
    else:
        alpha = uniform_weight_init(hidden_units, len(train_data[0]))
        beta = uniform_weight_init(len(train_label[0]), hidden_units + 1)

    alpha = np.array([[1, 1, 2, -3, 0, 1, -3], [1, 3, 1, 2, 1, 0, 2], [1, 2, 2, 2, 2, 2, 1], [1, 1, 0, 2, 1, -2, 2]])
    beta = np.array([[1, 1, 2, -2, 1], [1, 1, -1, 1, 2], [1, 3, 1, -1, 1]])
    x = np.array([1, 1, 1, 0, 0, 1, 1])
    y = np.array([0, 1, 0])
    o = NNforward(x, y, alpha, beta)
    print("OJ")
    print(o.J)
    [g_alpha, g_beta] = NNbackward(x, y, alpha, beta, o)
    print("alpha:")
    print(alpha)
    print("beta:")
    print(beta)
    alpha = alpha - 1 * g_alpha
    beta = beta - 1 * g_beta
    print("alpha:")
    print(alpha)
    print("beta:")
    print(beta)

    # for e in range(0, epochs):
    #     for i in range(0, len(train_data)):
    #         o = NNforward(train_data[i], train_label[i], alpha, beta)

    #         [g_alpha, g_beta] = NNbackward(train_data[i], train_label[i], alpha, beta, o)

    #         alpha = alpha - lr * g_alpha
    #         beta = beta - lr * g_beta
    # print("alpha:")
    # print(alpha.T)
    # print("beta:")
    # print(beta.T)

    # evaluate train mean cross-entropy
    # J_train = 0
    # for i in range(0, len(train_data)):
    #     o = NNforward(train_data[i], train_label[i], alpha, beta)
    #     J_train += o.J
    # with open(metrics_out, "a") as metrics_output:
    #     metrics_output.write("epoch=" + str(e + 1) + " crossentropy(train): " + str(J_train[0] / float(len(train_data))) + '\n')

    # # evaluate test mean cross-entropy
    # J_test = 0
    # for i in range(0, len(test_data)):
    #     o = NNforward(test_data[i], test_label[i], alpha, beta)
    #     J_test += o.J
    # with open(metrics_out, "a") as metrics_output:
    #     metrics_output.write("epoch=" + str(e + 1) + " crossentropy(validation): " + str(J_test[0] / float(len(test_data))) + '\n')

    return alpha, beta


def predict_all(test_data, alpha, beta):
    test_labels = []
    for i in range(0, len(test_data)):
        test_labels.append(predict(test_data[i], alpha, beta))
    return test_labels


def predict(x, alpha, beta):
    a = LinearForward(np.array(x).T, alpha)
    z = SigmoidForward(a)
    z_bias = np.array(np.append(1.0, z)).reshape(len(z) + 1, 1)
    b = LinearForward(z_bias, beta)
    y_hat = SoftmaxForward(b)
    return np.argmax(y_hat)


def uniform_weight_init(d0, d1):
    weights = np.random.uniform(-0.1, 0.1, (d0, d1))
    for weight in weights:
        weight[0] = 0
    return weights


def zeros_weight_init(d0, d1):
    return np.zeros((d0, d1))


def zeros_bias_init(d):
    return np.zeros(d)


def output(data, output_file):
    with open(output_file, "a") as output:
        for row in data:
            output.write(str(row) + '\n')


def calculate_error(predicted_labels, true_labels):
    # error value = (count of incorrect assigned labels)/(total data rows)
    count = 0
    for i in range(0, len(predicted_labels)):
        if predicted_labels[i] != np.argmax(true_labels[i]):
            count += 1
    return count / float(len(predicted_labels))


def read_data(input_file):
    labels = []
    data = []
    with open(input_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        label = np.zeros(10)
        label[int(line.split(',')[0])] = 1
        labels.append(label)
        data_item = [(float)(item) for item in line.split(',')[1:]]
        data_item.insert(0, 1)
        data.append(data_item)
    return np.array(data).reshape(len(data), len(data[0])), np.array(labels).reshape(len(labels), len(labels[0]))


if __name__ == "__main__":
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    epochs = (int)(sys.argv[6])
    hidden_units = (int)(sys.argv[7])
    init_flag = sys.argv[8]
    lr = (float)(sys.argv[9])

    train_data, train_labels = read_data(train_input)
    test_data, test_labels = read_data(test_input)
    alpha, beta = stocastic_gradient_descent(train_data, train_labels, hidden_units, test_data, test_labels, epochs, lr, init_flag, metrics_out)

    out_train_label = predict_all(train_data, alpha, beta)
    out_test_label = predict_all(test_data, alpha, beta)
    output(out_train_label, train_out)
    output(out_test_label, test_out)

    # error rate calculation
    with open(metrics_out, "a") as metrics_output:
        metrics_output.write("error(train): " + str(calculate_error(out_train_label, train_labels)) + '\n')
        metrics_output.write("error(validation): " + str(calculate_error(out_test_label, test_labels)) + '\n')
