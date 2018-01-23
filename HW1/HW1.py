
# coding: utf-8

import nltk
from nltk.corpus import stopwords

#get stopwords
stopWords = set(stopwords.words('english'))

#read target file
text_file = open('28.txt', 'r')
article = text_file.read()

#tokenization
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(article)
ps = nltk.stem.PorterStemmer()
tokens = []

#stopwords & stemming & lowercasing
for word in words:
    if word.lower() not in stopWords:
        tokens.append(ps.stem(word.lower()))

#write file
f = open('result.txt', 'w')
for token in tokens:
    f.write(token + '\n')
f.close()
