from scipy.spatial.distance import cdist
import logging

class KMeans1():
  # all array use below are ndarray, no comment!
  def __init__(self, n_clusters):
    self.n_clusters = n_clusters
    self.cluster_centers_ = None
    self.logger = logging.getLogger()
  def fit(self, X):
    # X: row data
    n, d = X.shape

    # random init labels
    Y = np.random.randint(0, self.n_clusters, n)

    # place holder center
    centers = np.zeros((self.n_clusters, d))

    count = 0
    while count<1000:
      self.logger.warning(f"------------------------\n\n")
      self.logger.warning(f"counter: {Counter(Y)}!")
      count+= 1
      # if count % 10 == 0:
      #   self.logger.warning(f"loop {count}!")
      # compute new center:
      for i in range(self.n_clusters):
        centers[i] = np.mean(X[Y==i], axis=0)
      
      # assign class
      ## compute distance to center
      distance_matrix = cdist(X, centers)
      ## assign new label
      Y_new = np.argmin(distance_matrix, axis=1)
      if (Y_new == Y).all():
        print(f"break after {count} iterations")
        break
      Y = np.copy(Y_new)

    self.cluster_centers_ = centers
    # return Y

  def predict(self, X):
    distance_matrix = cdist(X, self.cluster_centers_)
    labels = np.argmin(distance_matrix, axis=1)
    return labels

  @staticmethod
  def score(Y_true, Y_predict):
    # assume: max true
    # consider input order
    n_clusters = len(set(Y_true))
    Y_true_transform = np.zeros_like(Y_true)
    Y_predict_transform = np.copy(Y_predict)
    Y_predict_copy = np.copy(Y_predict)

    for i in range(n_clusters):
      yi = Y_true==i
      yi_pred = Y_predict_copy[yi]
      i_map = max(Counter(yi_pred).items(), key=lambda x: x[1] if x[0] >= 0 else -1)[0] # become positive ok!
      Y_true_transform[yi] = i_map
      Y_predict_copy[Y_predict_copy==i_map] = -1 # mark, change Y_predict here
    
    sc = accuracy_score(Y_true_transform, Y_predict_transform)
    return sc
