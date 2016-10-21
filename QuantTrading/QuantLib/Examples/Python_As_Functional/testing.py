# -*- coding: utf-8 -*-
from collections import defaultdict

def rank(data, key=lambda obj: obj):
    def rank_output(duplicates, key_iter, base=0):
        for k in key_iter:
            dups = len(duplicates[k])
            for value in duplicates[k]:
                yield (base+1+base+dups)/2, value
            base += dups
        
    def build_duplicates(duplicates, data_iter, key):
        for item in data_iter:
            duplicates[key(item)].append(item)
        return duplicates
    
    duplicates = build_duplicates(defaultdict(list), iter(data), key)
    return rank_output(duplicates, iter(sorted(duplicates)), 0)
    
ans = list(rank([0.8, 1.2, 1.2, 2.3, 18]))