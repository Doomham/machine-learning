import numpy as np
import time
import math
import random


class SVM(object):
    def __init__(self, train_x, train_y, test_x, test_y, C=200, gamma=10):
        self.train_x = np.mat(train_x)
        self.train_y = train_y
        self.test_x = np.mat(test_x)
        self.test_y = test_y
        self.C = C
        self.gamma = gamma
        self.alpha = [0] * len(self.train_y)
        self.b = 0
        self.KernelMatrix = self.calKernelMatrix()
        self.support_vec_index = []
        self.E = [self.cal_Ei(i) for i in range(len(self.alpha))]

    def calSingleKernel(self, xi, xj):
        z = np.mat(xi - xj)
        z = z * z.T
        d = 2 * (self.gamma ** 2)
        gaussian_kernel = np.exp(-z/d)
        return gaussian_kernel

    def calKernelMatrix(self):
        m = len(self.train_y)
        K = [[0 for i in range(m)] for j in range(m)]

        for i in range(m):
            for j in range(m):
                xi = self.train_x[i]
                xj = self.train_x[j]
                K[i][j] = self.calSingleKernel(xi, xj)

        return K

    def isSatisfyKKT(self, alpha, functional_margin, tol=0.001):
        #use tolerence
        y = functional_margin - 1
        condition1 = (abs(alpha) <= tol) and (y >= 0)
        condition2 = (0 < alpha) and (alpha < self.C) and (abs(y) < tol)
        condition3 = (abs(alpha - self.C) < tol) and (y < 0)

        if condition1:
            return True
        elif condition2:
            return True
        elif condition3:
            return True

        return False

    def g_xi(self, i):
        '''
        g(xi) = w.T * xi + b = sum(alpha[j] * label[j] * K(xj,xi)) j=1,...,m
        '''
        s = 0
        index = [i for i, alpha in enumerate(self.alpha) if (abs(alpha) > 0.0001)]   #only compute some of them
        for j in index:
            s += self.alpha[j] * self.train_y[j] * self.KernelMatrix[i][j]
        return s + self.b

    def cal_Ei(self, i):
        g_xi = self.g_xi(i)
        return g_xi - self.train_y[i]

    def find_alphaj(self, Ei):
        index = -1
        max_difference = -1
        is_not_zero = [i for i, E in enumerate(self.E) if E != 0]
        for j in is_not_zero:
            difference = abs(Ei - self.E[j])
            if difference > max_difference:
                index = j
                max_difference = difference
        if index == -1:
            index = int(random.uniform(0, len(self.E)))

        return index

    def calNew_alphaj(self, i, j, Ei, Ej):
        #compute the bounds L and H
        if self.train_y[i] == self.train_y[j]:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        else:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])

        #compute new_alphaj, then check∈[L, H]
        eta = self.KernelMatrix[i][i] ** 2 + self.KernelMatrix[j][j] ** 2 - 2 * self.KernelMatrix[i][j]
        alpha = self.alpha[j] + self.train_y[j] * (Ei - Ej) / eta
        if (alpha >= L) and (alpha <= H):
            return alpha
        elif alpha < L:
            return L
        elif alpha > H:
            return H

    def SMO(self, max_iter=3):
        '''
        use heuristic method to select alpha
        1.pick alphai(do not satisfy KKT), then compute Ei = g(xi) - yi
        2.search entire alpha, pick alphaj(maximize |Ei - Ej|)
        3.update alphai, alphaj, and b
        4.update Ei, Ej
        '''
        iteration = 0
        while iteration < max_iter:
            count = 0
            for i in range(len(self.alpha)):
                functional_margin = self.train_y[i] * self.g_xi(i)
                if self.isSatisfyKKT(self.alpha[i], functional_margin) == False:
                    Ei = self.E[i]
                    j = int(self.find_alphaj(Ei))
                    Ej = self.E[j]
                    alphaj_old = self.alpha[j]
                    alphai_old = self.alpha[i]
                    yi = self.train_y[i]
                    yj = self.train_y[j]
                    kii = self.KernelMatrix[i][i]
                    kjj = self.KernelMatrix[j][j]
                    kij = self.KernelMatrix[i][j]

                    self.alpha[j] = self.calNew_alphaj(i, j, Ei, Ej)
                    self.alpha[i] = alphai_old + yi * yj * (alphaj_old - self.alpha[j])
                    bi = -Ei - yi * kii * (self.alpha[i] - alphai_old) \
                         - yj * kij * (self.alpha[j] - alphaj_old) + self.b
                    bj = -Ej - yi * kij * (self.alpha[i] - alphai_old) \
                         - yj * kjj * (self.alpha[j] - alphaj_old) + self.b

                    if (self.alpha[i] > 0) and (self.alpha[i] < self.C):
                        self.b = bi
                    elif (self.alpha[j] > 0) and (self.alpha[j] < self.C):
                        self.b = bj
                    else:
                        self.b = (bi + bj) / 2

                    self.E[i] = self.cal_Ei(i)
                    self.E[j] = self.cal_Ei(j)
                else:
                    count += 1
            if count == len(self.alpha):
                break
            iteration += 1
        self.support_vec_index = [i for i, alpha in enumerate(self.alpha) if alpha > 0]

    def train(self):
        self.SMO()

    def test(self):
        count = 0
        for i in range(len(self.test_y)):
            pred_y = self.predicted_y(i)
            if pred_y == self.test_y[i]:
                count += 1
        acc = count / len(self.test_y)
        print('accuracy:', acc)
        print(self.test_y[:20], '| real number')
        print([int(self.predicted_y(i)) for i in range(20)], '| predicted number')

    def predicted_y(self, i):
        g = 0
        for j in self.support_vec_index:
            g += self.alpha[j] * self.train_y[j] * self.calSingleKernel(self.test_x[i],
                                                                        self.train_x[j])

        return np.sign(g + self.b)


def load_data(file_name):
    x = []
    y = []
    fr = open(file_name)
    for line in fr.readlines():
        current_line = line.strip().split(',')     #l[0]=label,str
        x.append([int(value)/255 for value in current_line[1:]])
        if int(current_line[0]) == 0:
            y.append(int(1))
        else:
            y.append(int(-1))
    return x, y


start = time.time()

train_x, train_y = load_data('mnist_train.csv')
test_x, test_y = load_data('mnist_test.csv')
svm = SVM(train_x[:1000], train_y[:1000], test_x[:500], test_y[:500])
svm.train()
svm.test()
print(len(svm.support_vec_index))
print('time span:', time.time() - start)

'''
accuracy: 0.98
[-1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1] | real number
[-1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1] | predicted number
support vectors:54
time span: 60.94188714027405

Process finished with exit code 0
'''

'''
SVM
我认为难度主要在SMO算法。人为设的C表征了SVM的软硬

1. 线性可分SVM
1.1 SVM做什么
    首先明确SVM是解决二分类问题的算法（也可以回归）。和其他的分类算法（如感知器，逻辑回归等）一样，它要找的是一个分开不同label数
    据的hyper plane。那么这个超平面和别的算法得到的平面有什么不一样？直观上说，它是一个使得两类label的样本都尽可能离他远的平面。
    要使得距离超平面最近的点也足够的远。
1.2 SVM的数学描述
    设最近的点距离超平面的几何间距（geometric margin）为γ，那么就是要使得γ最大。其他点到hyper plane的几何间距就有>=γ（这就是
    约束条件）。不难看出，这个问题是non-convex问题，因为约束条件不是一个凸集。于是两边同时乘||w||，右边看作一个整体这个就是凸集
    的约束了，记为γ*，同时目标函数改为γ*/||w||。
    很遗憾这问题依然是非凸的（目标函数非凸），注意到w和b同时乘一个数并不会改变hyper plane的位置，这是很关键的一步（将问题变成凸
    优化的问题），我们可以直接设γ*=1。这样就得到了书上的优化问题，即：
    max 0.5 * ||w||^2
    s.t. yi * (w.T * x + b) <= 1
    (CS231n从另一个角度hinge loss + l2正则)
1.3 优化问题的求解
1.3.1 写出原问题的拉格朗日形式
    这里的关键在于理解为什么max min L和原问题是一个意思。
'''

'''
凸优化？
什么是凸函数？
什么样的是凸集？
'''
