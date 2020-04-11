import  os
import re
import nltk
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
# nltk.download("all")
PATH_STOP_WORDS = r"./"


class PreProcessing:
    """
    pre processing data manual:
    """
    def __init__(self):
        self.SPECIAL_SIGNS = r"[~`@#\$\%\^\&\*\(\)\-_\+=\]\[{}\|\\\/<>\'\"\”\“,\‘\’\©\:\--.?;!\-\…]"
        self.lemmatizer = WordNetLemmatizer()
        with open(PATH_STOP_WORDS, 'r') as read:
            self.custom_stop_words = list(read.read().split(r'\n'))
        self.nltk_stop_words = stopwords.words("english")

    def remove_special_signs(self, text):
        return re.sub(self.SPECIAL_SIGNS, " ", self.text)

    def remove_spaces(self, text):
        return re.sub(r"\s+", " ", text)

    def remove_stop_words(self, text: str, customize=False):
        if customize:
            for word in self.custom_stop_wordsstop_words:
                text = text.replace(word, " ")
            return text
        else:
            for word in self.nltk_stop_words:
                text = text.replace(word, " ")
            return text

    def normalize(self, text):
        normal_text = self.remove_stop_words(text)
        normal_text = self.remove_special_signs(normal_text)
        normal_text = self.remove_spaces(normal_text)
        normal_text = normal_text.strip()
        return normal_text

    def lemmatizer(self, text):
        lemmatizer_text = ""
        for word in text.split():
            lemmatizer_text += self.lemmatizer.lemmatize(word)
        return  lemmatizer_text

    def tokenize(self, text):
        pass

    def create_dict_vocab(self, path_corpus):
        """
        :path_corpus: path to all data, normalized
        """
        all_words = set()
        all_files = []
        d_path = [path_corpus]
        while len(d_path) > 0:
            path = d_path.pop()
            for p in os.listdir(path):
                absolute_path = os.path.join(path_corpus, p)
                if os.path.isfile(absolute_path):
                    all_files.append(absolute_path)
                else:
                    d_path.append(absolute_path)
        for file in all_files:
            with open(file, 'r') as reader:
                text = reader.read()
                for word in text.split():
                    all_words.add(word)
        return list(all_words)
        # for (dirpath, dirnames, filenames) in walk(mypath):
        #     all_files.extend(filenames)
        #     break

 class Prepro():
    """using framework"""
    def __init__(self, min_df):
        self.stop_words = stopwords.words("english")
        self.min_df = min_df
        self.stemmer = PorterStemmer()

    def remove_stop_words(self, text):
        for word in self.stop_words:
                text = text.replace(" "+word+" ", " ")
        return text

    def get_words(self, text):
        return simple_preprocess(text)

    def stem(self, word):
      return self.stemmer.stem(word)

    def create_dict_from_array(self, doc): # not using
      """doc: list of list word"""
      dict_corpus = defaultdict(int)
      for d in doc:
          for w in d:
              dict_corpus[w] += 1
      return dict_corpus
    
    def get_data_from_dir(self, path):
        "return list of doc "
        all_files = []
        data = []
        for root, dirs, files in os.walk(path, topdown=True):
            for name in files:
                all_files.append(os.path.join(root, name))
    
        for file in all_files:
          with open(file, 'r') as reader:
            try:
              text = reader.read()
              data.append(text)
            except:
              pass
        return data

    def process(self, sentence):
      text = self.remove_stop_words(sentence)
      list_words = self.get_words(text)
      word_stem = [my_prepro.stem(w) for w in list_words]
      result = " ".join(word_stem)
    
    def save_serialize(obj, file_path):
      joblib.dump(obj, file_path)

    def create_word_freq(self, list_text):
      """ list text: list of paragraph"""
      word_freq = defaultdict(int) 
      word_doc_freq = defaultdict(int)
      n = len(list_text)
      for s in list_text:
        words = s.split()
        for word in words:
          word_freq[word] += 1
        set_words = set(words)
        for word in set_words:
          word_doc_freq[word] += 1
      for k in word_doc_freq.keys():
        word_doc_freq[k] /= n
      return word_freq, word_doc_freq