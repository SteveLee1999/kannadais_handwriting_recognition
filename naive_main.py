import numpy as np
from sklearn.model_selection import train_test_split
import handwritting_recognition as hr

if __name__ == '__main__':
    data_train = np.load("C:/Users/zhong/PycharmProjects/kannadais/data/X_kannada_MNIST_train.npz")['arr_0']
    label_train = np.load("C:/Users/zhong/PycharmProjects/kannadais/data/y_kannada_MNIST_train.npz")['arr_0']
    x_exam = np.load("C:/Users/zhong/PycharmProjects/kannadais/data/X_kannada_MNIST_test.npz")['arr_0']
    x_train, x_test, y_train, y_test = train_test_split(data_train, label_train, test_size=0.3, random_state=42, stratify=label_train)

    ii = 0
    n_x = 28 * 28
    n_h = 32
    n_y = 10
    alpha = 0.001
    parameters = hr.initialize_parameter(n_x, n_h, n_y)
    for i in range(60000):
        if i > 1000:
            alpha = alpha * 0.999
        img_train = x_train[i]
        label_train = np.zeros((10, 1))
        label_train[int(y_train[i])] = 1
        imgvector1 = hr.image2vector(img_train)
        imgvector = hr.normalize_data(imgvector1)

        A2, cache = hr.forward_propagation(imgvector, parameters)
        pre_label = hr.softmax(A2)
        costl = hr.calculate_loss(A2, label_train, parameters)
        grads = hr.back_propagation(parameters, cache, imgvector, label_train)
        parameters = hr.update_para(parameters, grads, learning_rate=alpha)
        grads["dW1"] = 0
        grads["dW2"] = 0
        grads["db1"] = 0
        grads["db2"] = 0
        print("cost after iteration %i:" % (i))
        print(costl)

    for i in range(18000):
        img_train = x_test[i]
        vector_image = hr.normalize_data(hr.image2vector(img_train))
        label_trainx = y_test[i]
        aa2, xxx = hr.forward_propagation(vector_image, parameters)
        predict_value = hr.softmax(aa2)
        if predict_value == int(label_trainx):
            ii = ii + 1
    print(ii)

    f = open("C:/Users/zhong/PycharmProjects/kannadais/data/result.txt", "w")
    for i in range(10000):
        img_test = x_exam[i]
        vector_image = hr.normalize_data(hr.image2vector(img_test))
        aa2, xxx = hr.forward_propagation(vector_image, parameters)
        predict_value = hr.softmax(aa2)
        f.write(str(predict_value))
