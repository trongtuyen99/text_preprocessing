import copy
class LogisticRegression1():
  
  def __init__(self, lr=0.05, lambd=1e-4, epochs=500):
    self.lr = lr
    self.lambd = lambd 
    self.epochs=epochs
    self.w = []

  def linear(self, W, X):
    """return (c, m)"""
    return np.dot(W.T, X) # Z
  
  def sigmoid(self, Z):
    """return (c, m)"""
    return 1/ (1+np.exp(-Z)) # A
  
  def grad(self, W, X, Y):
    Z = self.linear(W, X)
    A = self.sigmoid(Z)
    # return np.multiply((A-Y), X)
    return np.multiply((A-Y), X) + self.lambd * W

  
  def fit(self, X, Y, w_init=[]):
    d = X.shape[0]
    m = X.shape[1]
    if len(w_init) == 0:
      w_init = np.random.randn(d, 1)
    w0 = w_init
    all_idx = np.array([i for i in range(m)])
    loop = 0
    max_loop = self.epochs
    while(loop<max_loop):
      # start = time()
      all_idx = np.random.permutation(all_idx)
      for i in all_idx:
        xi = X[:, i].reshape(d, 1)
        yi = Y[i]
        # w_new = w0.reshape(d, 1) - self.lr * self.grad(w0, xi, yi).reshape(d, 1)
        # t0_grad = time()
        w_new = w0 - self.lr * self.grad(w0, xi, yi)
        # t1_grad = time()
        # print(f"compute grad after{t1_grad-t0_grad}")
        w0 = copy.deepcopy(w_new)
      end = time()
      # print(f"done loop {loop} after {end-start}s")
      loop += 1
    # self.w = w0
    return w0
  
  def fit_all(self, X, Y, w_init=[]):
    """
    Y: label, range(0, c)
    X, W: colmn matrix
    """
    X = np.array(X)
    Y = np.array(Y)
    classes = set(Y) # range(1, c)
    c = len(classes) # nof class
    d = X.shape[0] # dimension
    m = X.shape[1] # nof train data

    if len(w_init) == 0:
      w_init = np.random.randn(d, c)
    
    w = np.array([])
    # 1 vs rest.
    for idx in range(c):
      t1 = time()
      Y_train_tmp = Y == idx
      wc = self.fit(X, Y_train_tmp, w_init[:, idx].reshape(d, 1))
      # print("shape wc: ", wc.shape)
      w = np.append(w, [wc])
      # print(f"w[{idx}]: ", wc)
      t2 = time()
      print(f"done class {idx} after {t2-t1}s!\n")
    self.w = w.reshape(d,c, order='F')

  # def predict(self, X):
  #   Z = self.linear(self.w, X)
  #   A = self.sigmoid(Z)
  #   return A

  def predict(self, x):
    Z = np.dot(self.w.T, x)
    A = self.sigmoid(Z)
    # print(A)
    return np.argmax(A, axis=0)