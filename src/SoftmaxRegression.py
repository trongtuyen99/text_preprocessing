import numpy as np
from copy import deepcopy
# n: number of feature, m: number of data train
# c: number of class
# m: number of train data
# X(n+1, m): data train column matrix, (append row [1,1,1...] for b0)
# W(n+1, c) : weight
# Y(c, m): labels matrix, each column is one hot vector, sf labels for 1 observation
# Y_predict(c, m): predict labels, shape = shape(Y), each column is probality vector 
# ()

class SoftmaxRegression():
  def __init__(self, lr=0.05, tol=1e-4, max_iter=1000, print_after=100):
    self.lr = lr # learning rate
    self.tol = tol # tolerance
    self.max_iter = max_iter # max iteration
    self.print_after = print_after
    self.w = None
  
  def linear(self, W, X):
    """return (c, m)"""
    return np.dot(W.T, X) # Z
  
  def sigmoid(self, Z):
    """return (c, m)"""
    return np.exp(Z) # A
  
  def softmax(self, Z):
    """return (c, m)"""
    A = self.sigmoid(Z)
    return A / np.sum(A, axis=0)
  
  def softmax_new(self, Z):
    """return (c, m)"""
    Z_min = np.max(Z, axis=0, keepdims=True)
    Z_ = Z - Z_min
    A = self.sigmoid(Z_)
    return A / np.sum(A, axis=0)
  
  def cross_entropy(self, Y_predict, Y):
  # """return (1, m)"""
    return np.multiply(Y, np.log(Y_predict)) # * for elememtwise is ok too

  def cost(self, W, X, Y):
    """ return scalar"""
    Z = self.linear(W, X)
    Y_predict = self.softmax_new(Z)
    return -np.sum(self.cross_entropy(Y_predict, Y))

  def grad(self, W, X, Y):
    """dw return (n, c)"""
    Z = self.linear(W, X)
    # print("z: ", Z)
    Y_predict = self.softmax_new(Z)
    # print("predict: ", Y_predict)
    return np.dot(X, (Y_predict - Y).T)

  def fit(self, X, Y, w_init=[]):
    """X, Y: ndarray"""
    X = np.asarray(X)
    Y = np.asarray(Y)
    c = Y.shape[0] # number of class
    m = Y.shape[1] # no of observation
    n = X.shape[0] # number of feature
    idx = np.array([i for i in range(m)])
    if len(w_init) == 0:
      w_init = np.random.randn(X.shape[0], Y.shape[0])
    w = w_init
    loop = 0
    print_after = self.print_after
    while loop < self.max_iter:
      loop += 1
      np.random.shuffle(idx)
      for i in idx:
        x = X[:, i].reshape(n, 1)
        y = Y[:, i].reshape(c, 1)
        w1 = deepcopy(w)
        w -= self.lr * self.grad(w1, x, y) 
        if loop % print_after == 0:
          # print("cost: ", self.cost(w1, X, Y))
          if np.linalg.norm(w1-w) < self.tol:
            self.w = w
            return
    self.w = w
# batch, epochs, l2
  def predict(self, X):
    """X:column matrix"""
    X = np.asarray(X)
    Y_predict = []
    for x in X.T:
      # x = np.asarray(x).reshape(-1,1)
      z = self.linear(self.w, x)
      y = self.softmax_new(z)
      Y_predict.append(np.argmax(y))
    return Y_predict
  def predict_origin(self, x):
    z = self.linear(self.w, x)
    y = self.softmax_new(z)
    return np.argmax(y)