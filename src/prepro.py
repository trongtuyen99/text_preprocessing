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
    pre processing data:
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

    def bow_encoder(self, text):
        pass

    def tf_idf_encoder(self, text):
        pass