# -*- coding: utf-8 -*-
"""
Created on Fri May 23 10:32:40 2014

@author: gsubramanian
"""
import math
import copy
import operator
from collections import defaultdict

data =[2, 2, 3]

def FindOutlier(data):
    """Älgorithm to find outliers
    
    Args:
        Input data list
    
    """
    copy_data = copy.copy(data)
    data_entropy = entropy(copy_data)
    
    if entropy(copy_data) == 0:
        print "No Outliers found, data is homogeneous"
    else:
        entropy_dict = defaultdict(float)
        elements = set(copy_data)
        print "Individual Elements = ",elements
        for element in elements:
            copy_data = copy.copy(data)
            
            index_to_pop = [i for i in xrange(len(copy_data)) if element == copy_data[i]]
            trunc_data = [copy_data[i] for i in xrange(len(copy_data)) if i not in set(index_to_pop)]
                
            print "Data =",trunc_data
            entropy_value = entropy(trunc_data)
            entropy_increment =  data_entropy - entropy_value
            entropy_dict[element]=entropy_increment
            print "Removing, element = %d, Entropy = %f ,Entropy Decrement= %f"%(element,entropy_value,entropy_increment)
            
    
        sorted_entropy = sorted(entropy_dict.iteritems(),key=operator.itemgetter(1))
        
        if sorted_entropy[len(sorted_entropy)-1][1] == 0.0:
            print "No Outliers found, data is uniform"
        else:
            print "Outlier = %d "%(sorted_entropy[len(sorted_entropy)-1][0])

def prob(data,element):
    """Calculates the percentage count of a given element

    Given a list and an element, returns the elements percentage count

    Args:
        data        : List of element
        element     : Element of interest
    
    Returns:
        Probability of the elemnt in the list
        
    """
    element_count =0
    
    if len(data) == 0 or element == None \
                or not isinstance(element,(int,long,float)):
        return None
        
    element_count = data.count(element)

    return element_count / (1.0 * len(data))


def entropy(data):
    """"Calcuate entropy
    
    Args:
        data    : List for which entropy need to be calcuated
    
    Returns:
        Entropy of the given list
    """
    entropy =0.0
    
    if len(data) == 0:
        return None
    
    if len(data) == 1:
        return 0

    try:
        for element in data:
            p = prob(data,element)
            entropy+=-1*p*math.log(p,2)
    except ValueError as e:
        print e.message
        
    
    return entropy
    

if __name__ =="__main__":
    print "Given Data", data
    print "Entropy of the given list = %f"%entropy(data)
    FindOutlier(data)