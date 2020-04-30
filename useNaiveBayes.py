import numpy as np
import time

#load the training set or the testing set
#return x, y
def load_data(file_name):
    x = []
    y = []
    fr = open(file_name)
    for line in fr.readlines():
        current_line = line.strip().split(',')     #l[0]=label,str
        x.append([int(int(value) > 128) for value in current_line[1:]])
        y.append(int(current_line[0]))
    return x, y


#cal Py[k] = P(Y = k)
#Px_y[l][k][0] = P(X[l] = 0 | Y = k)
#Px_y[l][k][1] = P(X[l] = 1 | Y = k)
#for mnist,l = 28 * 28   k = 10,return = log(Py), log(Px_y)
#why log?since there are too many terms∈(0,1).
def calProb(train_x, train_y, class_num, features_num):

    Py = np.zeros(class_num)
    #cal sum of samples for every class k
    for i in range(len(train_y)):
        Py[train_y[i]] += 1


    Px_y = np.zeros((class_num, features_num, 2))
    #cal sum of pixels = 0/1 for every class
    for i in range(len(train_y)):   #ith sample
        for j in range(features_num):
            Px_y[train_y[i]][j][train_x[i][j]] += 1

    #cal Px_y(with laplacian smoot)
    Px_y += 1
    for k in range(class_num):
        Px_y[k][:][:] = (Px_y[k][:][:]) / (Py[k] + 2)
    #cal Py
    Py = (Py + 1)/(len(train_y) + class_num)
    log_Px_y = np.log(Px_y)
    log_Py = np.log(Py)
    return log_Py, log_Px_y


#input one test_x,return predicted y
def predict(x, log_Py, log_Px_y, class_num, features_num):
    log_prob = np.zeros(class_num)  #probability for every class
    for k in range(class_num):
        s = 0
        for feature in range(features_num):
            s += log_Px_y[k][feature][x[feature]]
        log_prob[k] = log_Py[k] + s
    return np.argmax(log_prob)


#calculate accuracy of testing set
def calAcc(test_x, test_y, log_Py, log_Px_y, class_num, features_num):
    correct_num = 0
    for item in range(len(test_y)):
        correct_num += float(
            int(predict(test_x[item], log_Py, log_Px_y, class_num, features_num)) == test_y[item])
    return correct_num/len(test_y)


start = time.time()
class_num = 10
features_num = 28 * 28

train_x, train_y = load_data('mnist_train.csv')
test_x, test_y = load_data('mnist_test.csv')

log_Py, log_Px_y = calProb(train_x, train_y, class_num, features_num)
acc = calAcc(test_x, test_y, log_Py, log_Px_y, class_num, features_num)

print('accuracy:', acc)

#output some predictions
print(test_y[:30], 'real numbers')
print([predict(test_x[i], log_Py, log_Px_y, class_num, features_num) for i in range(30)], 'predicted numbers')

print('time span:', time.time() - start)

'''
accuracy: 0.8433
[7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0, 6, 9, 0, 1, 5, 9, 7, 3, 4, 9, 6, 6, 5, 4, 0, 7, 4, 0, 1] real numbers
[7, 2, 1, 0, 4, 1, 4, 9, 4, 9, 0, 6, 9, 0, 1, 3, 9, 7, 3, 4, 9, 6, 6, 5, 4, 0, 7, 4, 0, 1] predicted numbers
time span: 101.4s
'''
