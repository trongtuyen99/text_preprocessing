def make_dict(corpus):
    coropa = {" ", "US0"} # " ": padding, US0: unseen word
    for sentence in corpus:
        for word in sentence.split():
            coropa.add(word)

    coropa = list(coropa)

    map_word = dict()
    for i, v in enumerate(coropa):
        map_word[v] = i
    return map_word

def corpus2vec(corpus, word2idx, sen_size):
    def sen2vec(sentence):
        s = sentence.split()
        tmp = list(map(lambda x: word2idx.get(x, word2idx.get("US0")), s))
        n = len(tmp)
        if n < sen_size:
            tmp += [word2idx[" "]] * (sen_size-n)
        
        vector_sentence = np.array(tmp[:80])
        return vector_sentence
    
    n = len(corpus)
    sentence_matrix = np.zeros((n, sen_size))

    for i in range(n):
        sentence_matrix[i] = sen2vec(corpus[i])
    return sentence_matrix

sentence_matrix = corpus2vec(X_train, word2idx, 80)