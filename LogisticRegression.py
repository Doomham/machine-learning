import numpy as np
import time


def load_data(file_name):
    x = []
    y = []
    fr = open(file_name)
    for line in fr.readlines():
        current_line = line.strip().split(',')     #l[0]=label,str
        x.append([int(value)/255 for value in current_line[1:]])
        if int(current_line[0]) == 0:
            y.append(1)
        else:
            y.append(0)
    return x, y


class Logistic_Regression(object):
    def __init__(self, train_file, test_file, lr=0.01):
        train_x, train_y = load_data(train_file)
        test_x, test_y = load_data(test_file)
        self.train_y = train_y
        self.test_y = test_y
        padding = np.mat([[1] * (len(self.train_y))]).T
        self.train_x = np.concatenate((padding, np.mat(train_x)), axis=1)
        padding = np.mat([[1] * (len(self.test_y))]).T
        self.test_x = np.concatenate((padding, np.mat(test_x)), axis=1)
        self.theta = np.mat(np.zeros((1, 28 * 28 + 1)))
        self.lr = lr

    def sigmoid(self, x):
        z = self.theta * x.T
        g = 1 / (1 + np.exp(-z))
        return g

    def cal_grad(self, i):
        grad = (self.train_y[i] - self.sigmoid(self.train_x[i, :])) * self.train_x[i, :]
        return grad

    def predict(self, y):
        if y >= 0.5:
            return 1
        else:
            return 0

    def train(self):
        for i in range(len(self.train_y)):
            self.theta[0, :] += self.lr * self.cal_grad(i)
            #更新Θ不要用loop
            if i % 50 == 0:
                print(self.test_y[:5], '|real number')
                print([self.predict(self.sigmoid(self.test_x[l])) for l in range(5)], '|predicted number')

    def test(self):
        error_count = 0
        for i in range(len(self.test_y)):
            pred_y = self.sigmoid(self.test_x[i])
            pred_y = self.predict(pred_y)
            if pred_y != self.test_y[i]:
                error_count += 1
        acc = 1 - (error_count / len(self.test_y))

        print(self.test_y[:20], '|real number')
        print([self.predict(self.sigmoid(self.test_x[l])) for l in range(20)], '|predicted number')
        print('accuracy:', acc)


start = time.time()

clf = Logistic_Regression('mnist_train.csv', 'mnist_test.csv')
clf.train()
clf.test()

print('time span:', time.time() - start)

'''
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0] |real number
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0] |predicted number
accuracy: 0.9922
time span: 25.438143014907837
'''

'''
Logistic Regression
关键在于认为P(y|x)已知，即：
P(Y = 1 | X = x) = h(x)
P(Y = 0 | X = x) = 1 - h(x)
合起来可以简写出：
P(Y = y | X = x) = (h(x))^y * (1 - h(x))^(1 - y)
接下来要应用极大似然估计，首先写出log-likelihhod（后面的各种求导也会轻松很多），然后求导得到梯度
得到梯度是一个求和，但我们采用随机地梯度下降即一个样本更新一次Θ
'''

'''
体会：
1.记得补上b和一列1， 拼接矩阵使用concatenate()
2.更新参数的时候切记不要用loop，太慢了
'''

'''
logistic regression用kernel trick怎么样？和svm的区别
'''
