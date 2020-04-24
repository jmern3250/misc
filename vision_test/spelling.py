import re
import numpy as np
from collections import Counter
import pickle
import copy
from Levenshtein import distance

with open('./data/names.p', 'rb') as f:
    ORG_NAMES, ALPH_NAMES = pickle.load(f)
    WORDS = Counter(ALPH_NAMES)
    CORRECTIONS = dict(zip(ALPH_NAMES, ORG_NAMES))

def correction(word):
    if word in ALPH_NAMES:
        name = word
    else:
        min_dist = np.inf
        for candidate in ALPH_NAMES:
            dist = distance(word, candidate)
            if dist < min_dist:
                name = candidate
                min_dist = dist
    try:
        name = CORRECTIONS[name]
    except:
        pass
    return name