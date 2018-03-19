
# coding: utf-8

# In[2]:


import os
# from sklearn.model_selection import train_test_split
import pickle
import math
import numpy as np
from math import *
from numpy.linalg import norm
import matplotlib.pyplot as plt


# # Part 2. Naive Bayes Algorithm

# Some helper functions

# In[2]:


# Obtaining data from the source file
# Split dataset into training 70%, validation 15%, and testing 15%.
def get_datasets(file):
    
    # files = os.listdir(data_dir)
    # for file in file_list:
    X = []
    y = []
    
    if file.split('.')[0][-4:] == 'fake':
        y_l = 1     # Fake = 1, Real = 0
    else:
        y_l = 0
        
    file = open(file, 'r').read().split('\n')
    for line in file:
        X.append(line)
        y.append(y_l)
    # Splitting the sets
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.30, seed=1)
    (X_test, X_val, y_test, y_val) = train_test_split(X_test, y_test, test_size=0.50, seed=1)

    return X_train, y_train, X_test, y_test, X_val, y_val

def train_test_split(x,y,test_size=0,seed=411):
    np.random.seed(seed)
    x_test, x_train = x[:int(len(x)*test_size)], x[int(len(x)*test_size):]
    y_test, y_train = y[:int(len(y)*test_size)], y[int(len(y)*test_size):]
    return x_train, x_test, y_train, y_test

def save_data(data, name):
    with open(name + '.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def load_data(name):
    with open(name + '.pickle', 'rb') as handle:
        saved_data = pickle.load(handle)
    return saved_data

def build_LM(dataset):
    '''
    Creating dictionary that has a word as key, then a value that is the count of the word
    '''
    LM = {}
    # Adding counts
    for sentence in dataset:
        sentence = sentence.split()
        for word in sentence:
            if word not in LM.keys():
                LM[word] = 1
            else:
                LM[word] += 1
    return LM

def prob(sentence, LM, m, p):    
    '''
    Input: sentence: process sentence, string
    LM: Unigram model created by function build_LM
    m, p: parameters
    '''
    sentence = sentence.split()
    total = sum(LM.values())
    log_mle = 0
    for w in sentence:
        if w not in LM.keys():
            n = m*p
            d = total + m
            log_prob = math.log(n/d)
        else:
            n = LM[w] + m*p
            d = total + m
            log_prob = math.log(n/d)
        log_mle += log_prob
    for w in LM.keys():
        if w not in sentence:   # We have already included the case when w in sentence
            n = LM[w] + m*p
            d = total + m
            log_prob = math.log(1-n/d)
        log_mle += log_prob
    prob = math.exp(log_mle)
    return prob

def accuracy(X_test, y_test, fake, real, m, p):
    '''
    inputs:
        fake: Language model built in build_LM
        real: Language model built in build_LM
        X_test, y_test: test/validation cases and label
    return:
        accuracy of the testing cases
    '''
    count = 0.0
    pf = float(len(fake.keys()))/float(len(fake.keys())+len(real.keys()))
    pr = float(len(real.keys()))/float(len(fake.keys())+len(real.keys()))    

    for i in range(len(X_test)-1):
        fake_prob = prob(X_test[i], fake, m, p)*pf
        real_prob = prob(X_test[i], real, m, p)*pr
    
        if fake_prob > real_prob:
            predict = 1
        else:
            predict = 0
        if predict == y_test[i]:
            count += 1
    return count/len(X_test)
  


# In[3]:


def naive_bayes(fX_train, fy_train, X_val, y_val, X_test, y_test):
    # Building dictionaries that contains the Unigram informations, for each key 'word', the value is the count of word appearance
    fake = build_LM(fX_train)
    real = build_LM(rX_train)
 
    # Saving data
    save_data(fake, 'fake')
    save_data(real, 'real')
    load data
    fake = load_data('fake')
    real = load_data('real')
    
    p = 0.000001
    m = 150
    X_train = fX_train + rX_train
    y_train = fy_train + ry_train
    X_val = fX_val + rX_val
    y_val = fy_val + ry_val
    X_test = fX_test + rX_test
    y_test = fy_test + ry_test
    
    a_train = accuracy(X_train, y_train, fake, real, m, p)
    a_val = accuracy(X_val, y_val, fake, real, m, p)
    a_test = accuracy(X_test, y_test, fake, real, m, p)
    
    print 'p = ' + str(p)
    print 'm = ' + str(m)
    print('Accuracy on training set is '+ str(a_train))
    print('Accuracy on validation set is '+ str(a_val))
    print('Accuracy on test set is '+ str(a_test))
   
    # Results
    # p = 0.02
    # m = 150
    # Accuracy on training set is 0.917796239615
    # Accuracy on validation set is 0.859470468432
    # Accuracy on test set is 0.855102040816
    # p = 0.01
    # m = 300
    # Accuracy on training set is 0.919545255794
    # Accuracy on validation set is 0.859470468432
    # Accuracy on test set is 0.85306122449


# In[5]:


fake_data = '/Users/yingxue_wang/Documents/cdf/csc411/Project3/data/clean_fake.txt'
real_data = '/Users/yingxue_wang/Documents/cdf/csc411/Project3/data/clean_real.txt'
fX_train, fy_train, fX_test, fy_test, fX_val, fy_val = get_datasets(fake_data)
rX_train, ry_train, rX_test, ry_test, rX_val, ry_val = get_datasets(real_data)

X_train = fX_train + rX_train
y_train = fy_train + ry_train
X_val = fX_val + rX_val
y_val = fy_val + ry_val
X_test = fX_test + rX_test
y_test = fy_test + ry_test

# naive_bayes(fX_train, fy_train, X_val, y_val, X_test, y_test)


# # Part 3. Picking Relevant/Non-relevant 

# # Part 4: Logistic Regression Algorithm

# Some helper functions

# In[20]:


def get_mapped_matrix(X_train, word_list):
    c = 0.0
    X_matrix = np.ones((len(word_list),1))
    for data in X_train:
        c += 1
        if c%100 == 0:
            print c
        x = np.zeros((len(word_list),1))
        data = data.split()
        for i in range(len(word_list)):
            if word_list[i] in data:
                x[i] = 1
        X_matrix = np.hstack((X_matrix, x))
    X_matrix = X_matrix[:,1:] 
    return X_matrix


# In[27]:


def sigmoid(y):
    # return 1/(1+np.exp(-(np.dot(theta.T, x))))
    return 1/(1+np.exp(-y))
  
def f(x, y, theta):
    '''
    output dimenssion:(2287,)
    '''
    x = np.vstack((np.ones((1, len(x[0]))), x))
    return y*np.log(sigmoid(np.dot(theta.T,x))) + (1-y)*np.log(1-sigmoid(np.dot(theta.T,x)))

def df(x, y, theta):
    # (4821,)
    x = np.vstack((np.ones((1, len(x[0]))), x))
    return np.sum((sigmoid(np.dot(theta.T,x))-y)*x, axis = 1)


def grad_descent(f, df, x, y, init_t, alpha):
    thetas = np.empty((x.shape[0]+1))
    EPS = 1e-5   
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 4000
    iter = 0
    while norm(t - prev_t) > EPS and iter < max_iter:  
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        iter += 1
        if iter % 50 == 0:
            thetas = np.vstack((thetas,t))
            print iter
    return thetas


# In[23]:


def performace_test(X_test, y_test, thetas):
    performance = []
    
    X_test = np.vstack((np.ones((1, len(X_test[0]))), X_test))
    for theta in thetas:
        count = 0.0
        predict = sigmoid(np.dot(theta.T, X_test))
        for i in range(len(predict)):
            if predict[i] > 0.5:
                predict[i] = 1
            else:
                predict[i] = 0
            if predict[i] == y_test[i]:
                count += 1
        performance.append(count/X_test.shape[1])
    return performance
     


# In[30]:


# Obtainning input matrix using the full training set
# LM = build_LM(X_train)
# word_list = list(LM.keys())     # len(word_list) = 4820
# X_train = get_mapped_matrix(X_train, word_list)
# X_val = get_mapped_matrix(X_val, word_list)
# save_data(X_train, 'X_train')
# save_data(X_val, 'X_val')

X_train = load_data('X_train')
X_val = load_data('X_val')

# print X_train.shape    # (4820, 2287)

# Number of features: 4820 + 1 bias -> X:(4820, 2287)
# theta : (4821,)
# theta.T * x : 1 * 2287
# y = 1 * 2287
theta = np.zeros((4821))
y_train = np.reshape(y_train,(X_train.shape[1])).T     # (2287,)
alpha = 0.0001

# Plot learning curves: Performance versus iterations
thetas = grad_descent(f, df, X_train, y_train, theta, alpha)
save_data(thetas, 'thetas')


# In[31]:


thetas = load_data('thetas')

performance_val = performace_test(X_val, y_val, thetas)
performance_train = performace_test(X_train, y_train, thetas)
save_data(performance_val, 'performance_val')
save_data(performance_train, 'performance_train')
print performance_val
print performance_train


# Plot the Learning curve

# In[35]:


# Plot performance on validation set
# Plot performance on training set
performance_val = load_data('performance_val')
performance_train = load_data('performance_train')
iteration = range(0,4001,50)
plt.plot(iteration, performance_val, iteration, performance_train)
plt.legend('Validation set', 'Training set')
plt.xlabel('Iteraion')
plt.ylabel('Performance')
plt.savefig("performance_4000_0_0001.png")
plt.show()


# # Part 7. Decision Tree

# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
