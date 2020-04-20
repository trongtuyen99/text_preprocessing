from collections import Counter
class MuitinomialNB1():

  def __init__(self):
    self.pc = None
    self.pxc = None
  def fit(self, X, Y):
    """X: ndarray countvectorize (row vector), Y: ndarray label"""
    def compute_pc(Y):
      n = len(Y)
      c = Counter(Y)
      for k in c.keys():
        c[k] /= n
      return c

    self.pc = compute_pc(Y)

    def compute_pxc(X, Y):
      # measure ndarray type
      X = np.asarray(X)
      Y = np.array(Y)
      row, col = X.shape[1], len(self.pc)
      pxc = np.zeros((row, col))
      for c in range(col):
        idx = Y==c
        # print("idx[:10]: ", idx[:10])
        n = sum(idx) # size of class c
        x = X[idx]
        for i in range(row):
          # xi = sum(x[:][i]>0)
          xi = sum(x[:, i])
          pxic = xi/n
          pxc[i][c] = pxic
      return pxc
    self.pxc = compute_pxc(X, Y)
  def predict(self, x): # x: countVectorize
    n = len(self.pc)
    prob = np.zeros(n)
    for i in range(n):
      p = self.pc[i]
      for j in range(len(x)):
        if x[j] > 0:
          p *= self.pxc[j][i]
      prob[i] = p
    return np.argmax(prob)
