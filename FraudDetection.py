# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:44:14 2014
A toy example of Fraud detection using Entropy measure.

The data is 

Tid     Spent (INR)     Paid (INR)      Country     Balance
1       1000            200             1           800
2       900             200             1           1500
3       1000            190             1           2310
4       7200            200             2           9310


@author: gsubramanian
"""

import numpy as np
from Entropy import entropy

# data
data = np.matrix([[1000,200,1],[900,200,1],[1000,190,1],[7200,200,2]])
rows,cols = data.shape



def getEntropy(input_data):
    """Calcuate entropy of the whole dataset
    """
    print input_data
    entropySum=0
    rows,cols = input_data.shape
    for i in xrange(cols):
        inData=[item[0] for item in input_data[:,i].tolist()]
        entropySum+=entropy(inData)
    return entropySum
   

if __name__ == "__main__":

    data_entropy = getEntropy(data)
    print "Whole Data Entropy = ",data_entropy
    
    
    mask = np.ones(rows ,dtype=bool)
    
    for i in xrange(rows):
        mask[[i]] = False
        print mask
        in_data =data[mask]
        ent = getEntropy(in_data)
        ent_decrease = data_entropy - ent
        print "Removing record = %d , Entropy = %f , Decrease = %f "%(i,ent,ent_decrease)
        mask[[i]] = True
    


