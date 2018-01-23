
# coding: utf-8

# In[8]:


import nltk
import nltk.corpus as corpus
import numpy as np
from collections import Counter
import math
# get stopwords
stopWords = set(corpus.stopwords.words('english'))


# In[9]:


doc_num = 1095
df_dict = {}
index_dict = {}
doc_words = []
doc_path = 'IRTM/'
tf_path = 'vector/'


# In[ ]:


# stemming & remove stop words & count df
ps = nltk.stem.PorterStemmer()
tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z]+')
for d in range(doc_num):
    temp_dict = {}
    doc_file = open(doc_path + str(d+1) + '.txt', 'r')
    article = doc_file.read()
    words = tokenizer.tokenize(article)
    for w in words:
        temp = str(ps.stem(w.lower()))
        if temp not in stopWords:
            if temp not in temp_dict:
                temp_dict[temp] = 1
            else:
                 temp_dict[temp] += 1
    for k in temp_dict.keys():
        if k in df_dict:
            df_dict[k] += 1
        else:
            df_dict[k] = 1
    doc_words.append(temp_dict)
    doc_file.close()


# In[ ]:


# sort dictionary by keys
sorted_dict = [(k, df_dict[k]) for k in sorted(df_dict.keys())]
# write dictionary to file
dict_file = open('dictionary.txt', 'w')
t_index = 1
for k, v in sorted_dict:
    index_dict[k] = t_index
    s = str(t_index) + ' ' + k + ' ' + str(v) + '\n'
    dict_file.write(s)
    t_index += 1
dict_file.close()


# In[ ]:


for d in range(doc_num):
    # get tf
    words = doc_words[d]
    tf = Counter(words)
    tf = [ (index_dict[k] ,tf[k]*math.log(doc_num/df_dict[k], 10))
          for k in sorted(tf.keys())]

    # calculate length for unit vector
    v_len = 0
    for n in tf:
        v_len += n[1]**2
    v_len = v_len**0.5
    tf = [ (i[0], i[1]/v_len) for i in tf]

    # write files
    tf_file = open(tf_path + str(d+1) + '.txt', 'w')
    tf_file.write(str(len(tf)) + '\n')
    for n in tf:
        s = str(n[0]) + ' ' + str(n[1]) + '\n'
        tf_file.write(s)
    tf_file.close()


# In[ ]:


# cosine function
def cosine(Docx, Docy):
    x_vector = np.zeros(len(df_dict))
    y_vector = np.zeros(len(df_dict))
    with open(tf_path + str(Docx) + '.txt', 'r') as x_file:
        lines = 0
        terms = 0
        for line in x_file:
            if lines != 0:
                temp = line.split(' ')
                x_vector[int(temp[0])] = float(temp[1])
            else:
                terms = line
            lines += 1
    with open(tf_path + str(Docy) + '.txt', 'r') as y_file:
        lines = 0
        terms = 0
        for line in y_file:
            if lines != 0:
                temp = line.split(' ')
                y_vector[int(temp[0])] = float(temp[1])
            else:
                terms = line
            lines += 1
    return (x_vector * y_vector).sum()
