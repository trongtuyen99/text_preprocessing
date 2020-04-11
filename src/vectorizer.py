class CounterVectorize():
  def __init__(self, path_word_freq, df):
    self.word_freq = joblib.load(path_word_freq)
    self.df = df # for using word that freq > df
    all_words = list(filter(lambda x: x[1]>df, self.word_freq))
    self.all_words = list(dict(all_words).keys())
  def vectorize(self, sentence):
    words = sentence.split()
    n = len(self.all_words) + 1
    vector = [0] * n
    for w in words:
      flag = False
      for i, w2 in enumerate(self.all_words):
        if w == w2:
          vector[i] += 1
          flag = True
      if not flag:
        vector[n-1] += 1 # word not exist in all_words
    return vector
  def count_vectorize(self, list_text):
    rs = []
    for s in list_text:
      rs.append(vectorize(s))
    return rs

class TfIdfVectorize():
  def __init__(self, path_word_doc_freq, path_word_freq, df):
    self.df = df
    word_doc_freq = joblib.load(path_word_doc_freq)
    self.word_doc_freq = dict(word_doc_freq)

    self.word_freq = joblib.load(path_word_freq)
    all_words = list(filter(lambda x: x[1]>df, self.word_freq))
    self.all_words = list(dict(all_words).keys())
  def vectorize(self, sentence):
    words = sentence.split()
    n = len(self.all_words) + 1
    vector = [0] * n
    def tf(words):
      n = len(words)
      w_freq = Counter(words) # vectorize here is ok
      for k in w_freq.keys():
        w_freq[k] /= n
      return w_freq 
    def idf(words):
      import math
      n = len(words)
      rs = defaultdict()
      for w in words:
        r = self.word_doc_freq.get(w, 1)
        rs[w] = math.log(1/r) # inverse doc freq
      return rs

    tf_ = tf(words)
    idf_ = idf(words)
    for w in words:
      tf_idf = tf_[w] * idf_[w]
      flag = False
      for i, w2 in enumerate(self.all_words):
        if w == w2:
          vector[i] += tf_idf
          flag = True
      if not flag:
        vector[n-1] += 1 # word not exist in all_words
    return vector

  def tfidf_vectorize(self, list_text):
    rs = []
    for sentence in list_text:
      pass
