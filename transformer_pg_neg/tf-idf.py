


import jieba
import jieba.posseg as pseg
import os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer


with open('./data/test_x.txt','r',encoding='utf-8') as f:
    data_x = f.readlines()

corpus = []
for line in data_x:
    line = line.strip()
    corpus.append(line)

cv = CountVectorizer()
cv_fit=cv.fit_transform(corpus)

# print(cv.get_feature_names())
print(cv_fit.toarray()[0])









