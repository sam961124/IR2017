
# coding: utf-8

# In[1]:


import nltk
import nltk.corpus as corpus
import numpy as np
import pandas as pd
# get stopwords
stopWords = set(corpus.stopwords.words('english'))

classes = 13
doc_path = 'IRTM/'


# In[2]:

print("Loading Files...")
doc_class = []
training_file = pd.read_table('training.txt', header=None)
for i in range(classes):
    temp = training_file[0][i].split(' ')
    doc_class.append(temp[1:-1])
doc_class = np.array(doc_class)
training_doc_flat = doc_class.flatten()

# split training & valid set
valid_labels = []
valid_size = 0
train_doc = doc_class[:,:15-valid_size]
valid_doc = doc_class[:,15-valid_size:]
for c in range(classes):
    for i in range(valid_size):
        valid_labels.append(c+1)
print("File Loaded!")

# In[3]:


def ExtractVocabulary(doc):
    temp_dict = {}
    ps = nltk.stem.PorterStemmer()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z]+')
    count = 1
    for i in range(len(doc)):
        doc_file = open(doc_path + doc[i] + '.txt', 'r')
        article = doc_file.read()
        words = tokenizer.tokenize(article)
        for w in words:
            temp = str(ps.stem(w.lower()))
            if temp not in stopWords:
                if temp not in temp_dict:
                    temp_dict[temp] = count
                    count += 1
    return(temp_dict)

def BuildFeaturesLabels(doc_class, vocab):
    labels = np.zeros(195)
    features = np.zeros((195, len(vocab)+1))
    count = 0
    for c in range(classes):
        for d in doc_class[c]:
            labels[count] = c+1
            doc_vocab = ExtractVocabulary([d])
            terms = []
            for v in doc_vocab:
                features[count][vocab[v]] = 1
            count += 1
    return features, labels
    
def ComputeChi(features, labels):
    N = len(labels)
    term_chi = []
    for t in range(features.shape[1]):
        temp = 0
        for c in range(classes):
            start_idx = c*15
            end_idx = (c+1)*15
            presents = features[:,t].sum()
            absents = N - presents
            on_topic_present = features[start_idx:end_idx,t].sum()
            off_topic_present = presents - on_topic_present
            on_topic_absent = 15 - on_topic_present
            off_topic_absent = absents - on_topic_absent
            E = N * (on_topic_present + off_topic_present)/N * (on_topic_present + 
                                                                on_topic_absent)/N
            temp += (on_topic_present - E)**2/E
        term_chi.append(temp/classes)
    return term_chi
            
    
def SelectFeatures(all_vocab, features, labels, num):
    chi_list = ComputeChi(features, labels)
    index = np.argsort(chi_list)
    selected_vocab = {k: v for k, v in all_vocab.items() if v in index[:num]}
    return(index[-num-1:-1])


# In[4]:


def ConcatTextInClass(doc_class, c, vocab):
    text = []
    ps = nltk.stem.PorterStemmer()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z]+')
    for d in doc_class[c]:
        doc_file = open(doc_path + d + '.txt', 'r')
        article = doc_file.read()
        words = tokenizer.tokenize(article)
        for w in words:
            temp = str(ps.stem(w.lower()))
            if temp not in stopWords and temp in vocab:
                text.append(temp)
    return text

def TrainMultinomialNB(train_doc, vocab):
    N = 195 - classes * valid_size
    prior = np.zeros(classes)
    condprob = np.zeros((len(vocab), classes))
    class_count = np.unique(labels, return_counts=True)
    for c in range(classes):
        T = np.zeros(len(vocab))
        Nc = 15 - valid_size
        prior[c] = Nc / N
        text_c = ConcatTextInClass(train_doc, c, vocab)
        for t in range(len(vocab)):
            count = 0
            for v in text_c:
                if v == vocab[t]:
                    count += 1
            T[t] = count
        for t in range(len(vocab)):
            condprob[t][c] = (T[t] + 1) / (len(text_c) + len(vocab))
    return prior, condprob

def ExtractTokensFromDocs(vocab, d):
    text = []
    ps = nltk.stem.PorterStemmer()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z]+')
    doc_file = open(doc_path + d + '.txt', 'r')
    article = doc_file.read()
    words = tokenizer.tokenize(article)
    for w in words:
        temp = str(ps.stem(w.lower()))
        if temp not in stopWords and temp in vocab:
            text.append(vocab.index(temp))
    return text

def ApplyMultinomialNB(vocab, prior, condprob, d):
    W = ExtractTokensFromDocs(vocab, d)
    score = np.zeros(classes)
    for c in range(classes):
        score[c] = np.log(prior[c])
        for t in W:
            score[c] += np.log(condprob[t][c])
    return np.argmax(score) + 1


# In[5]:

print("Selecting Features...")
# feature selection
all_vocab = ExtractVocabulary(training_doc_flat)
features, labels = BuildFeaturesLabels(doc_class, all_vocab)
selected_index = SelectFeatures(all_vocab, features, labels, 500)
selected_vocab = [k for k, v in all_vocab.items() if v in selected_index]
print("Features Selected!")

# In[6]:


# train NB
print("Start Training...")
prior, condprob = TrainMultinomialNB(train_doc, selected_vocab)
print("Training Finished!")

# In[7]:


# output
print("Start Predicting...")
out_file = "r06725015.txt"
all_index = np.arange(1,1096,1)
train_index = np.array(train_doc.flatten(), dtype=int)
output_index = list(set(all_index).difference(set(train_index)))
text_file = open(out_file, "w")
for d in output_index:
    res = ApplyMultinomialNB(selected_vocab, prior, condprob, str(d))
    text_file.write(str(d) + '\t' + str(res) + '\n')
text_file.close()
print("Output Prediction as " + out_file + "!")