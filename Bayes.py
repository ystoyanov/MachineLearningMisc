# -*- coding: utf-8 -*-
"""
Created on Fri May 23 13:55:39 2014


A sample program to demonstrate the principles
of naive bayes to identify gender of the given
first name.

@author: gsubramanian
"""

from nltk.corpus import names
from collections import defaultdict

# Data
mnames =[(name,'M') for name in names.words('male.txt')]
fnames =[(name,'F') for name in names.words('female.txt')]

male_dict = defaultdict(int)
female_dict = defaultdict(int)


def gen_features(word):
    """"Get the last character
        Given a word, return the last character    
    """
    if word == None: return None    
    
    return 'last_letter:' + word[-1]
    

def train_model():
    """"Build a model to predict the gender
    """
    for data in mnames:
        feature = gen_features(data[0])
        male_dict[feature]+=1
    
    
    for data in fnames:
        feature = gen_features(data[0])
        female_dict[feature]+=1


def predict_name(word):
    """"Given a word predict the gender
    """
    feature = gen_features(word)
    male_entries = male_dict[feature]
    female_entries = female_dict[feature]
    total_entries = male_entries + female_entries
    
    if (male_entries / (1.0*total_entries)) > (female_entries/(1.0*total_entries)):
        print "Given word=%s is a male"%(word)
    else:
        print "Given word=%s is a female"%(word)


if __name__ =="__main__":
    print 'Train Model'
    train_model()
    predict_name('model')
            