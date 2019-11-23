

import pickle
import numpy as np
def load_obj():
    with open('./data/frequence_neg.pkl', 'rb') as f:
        return pickle.load(f)
fre = load_obj()
print(min(fre.values()))
a = min(fre.values())
word_freqs = np.array(list(fre.values()))
print(fre['广告'])
# print(fre.get('维修',a)/word_freqs.sum())






