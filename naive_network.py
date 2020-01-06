import numpy as np


def normalize_data(ima):
    a_max = np.max(ima)
    a_min = np.min(ima)
    for j in range(ima.shape[0]):
        ima[j] = (ima[j] - a_min) / (a_max - a_min)
    return ima


# 初始化参数，权重
# w1.shape = n_h*n_x, b1.shape = n_h*1, w2.shape = n_y*n_h, b1.shape = n_y*1
# 返回一个字典
def initialize_parameter(n_x, n_h, n_y):
    np.random.seed(2)
    w1 = np.random.uniform(-np.sqrt(6) / np.sqrt(n_x + n_h), np.sqrt(6) / np.sqrt(n_h + n_x), size=(n_h, n_x))
    b1 = np.zeros((n_h, 1))
    w2 = np.random.uniform(-np.sqrt(6) / np.sqrt(n_y + n_h), np.sqrt(6) / np.sqrt(n_y + n_h), size=(n_y, n_h))
    b2 = np.zeros((n_y, 1))
    parameters = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    return parameters


# 前向传播
# x.shape = n_x*n_y
def forward_propagation(x, parameters):
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    z1 = np.dot(w1, x) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)
    ret = {"z1": z1, "a1": a1, "z2": z2, "a2": a2}
    return a2, ret


# 反向传播
def back_propagation(parameters, cache, x, y):
    w1 = parameters["w1"]
    w2 = parameters["w2"]
    a1 = cache["a1"]
    a2 = cache["a2"]
    z1 = cache["z1"]
    dz2 = a2 - y
    dw2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.dot(w2.T, dz2) * (1 - np.power(a1, 2))
    dw1 = np.dot(dz1, x.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)
    grads = {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2}
    return grads


def update_para(parameters, grads, learning_rate):
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    dw1 = grads["dw1"]
    db1 = grads["db1"]
    dw2 = grads["dw2"]
    db2 = grads["db2"]
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2
    parameters = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    return parameters


# 计算损失
def calculate_loss(a2, y, parameters):
    t = 0.00000000001
    logprobs = np.multiply(np.log(a2 + t), y) + np.multiply(np.log(1 - a2 + t), (1 - y))
    cost = np.sum(logprobs, axis=0, keepdims=True) / a2.shape[0]
    return cost


def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s


def image2vector(image):
    v = np.reshape(image, [784, 1])
    return v


def softmax(x):
    v = np.argmax(x)
    return v
