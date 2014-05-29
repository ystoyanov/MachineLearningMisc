# -*- coding: utf-8 -*-
"""
Created on Thu May 29 16:19:57 2014

@author: gsubramanian
"""
from sklearn import tree

x = [[1000,200,1],[900,200,1],[1000,190,1],[7200,200,2]]
y=[0,0,0,1]

clf = tree.DecisionTreeClassifier()
clf.fit(x,y)

print clf.predict([[200000,400,1]])


