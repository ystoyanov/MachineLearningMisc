"""
Demonstrate how to build a quick and dirty
Base Line Model using scikit learn

"""

# Load Libraries
import numpy as np 
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix


# Let us use Iris dataset
iris = datasets.load_iris()
x = iris.data
y = iris.target


from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='stratified', random_state = 100, constant = None)
dummy.fit(x, y)


## Number of classes
print "Number of classes :{}".format(dummy.n_classes_)
## Number of classes assigned to each tuple
print "Number of classes assigned to each tuple :{}".format(dummy.n_outputs_)
### Prior distribution of the classes.
print "Prior distribution of the classes {}".format(dummy.class_prior_)


output = np.random.multinomial(1, [.33,.33,.33], size = x.shape[0])
predictions = output.argmax(axis=1)

print output[0]
print predictions[1]


y_predicted = dummy.predict(x)

print y_predicted
# Find model accuracy
print "Model accuracy = %0.2f"%(accuracy_score(y,y_predicted) * 100) + "%\n"

# Confusion matrix
print "Confusion Matrix"
print confusion_matrix(y, y_predicted, labels=list(set(y)))


############ strategy : constant #########################

### Let us create a imbalanced dataset.
from sklearn.datasets import make_classification

x, y = make_classification(n_samples = 100, n_features = 10, weights = [0.3, 0.7], random_state = 100)
# Print the class distribution
print np.bincount(y)

dummy1 = DummyClassifier(strategy='stratified', random_state = 100, constant = None)
dummy1.fit(x, y)
y_p1 = dummy1.predict(x)

dummy2 = DummyClassifier(strategy='most_frequent', random_state = 100, constant = None)
dummy2.fit(x, y)
y_p2 = dummy2.predict(x)


from sklearn.metrics import precision_score, recall_score, f1_score
print
print "########## Metrics #################"
print "     Dummy Model 1, strategy: stratified,    accuracy {0:.2f}, precision {0:.2f}, recall {0:.2f}, f1-score {0:.2f}"\
						.format(accuracy_score(y, y_p1), precision_score(y, y_p1), recall_score(y, y_p1), f1_score(y, y_p1))
print "     Dummy Model 2, strategy: most_frequent, accuracy {0:.2f}, precision {0:.2f}, recall {0:.2f}, f1-score {0:.2f}"\
						.format(accuracy_score(y, y_p2), precision_score(y, y_p2), recall_score(y, y_p2), f1_score(y, y_p2))

print


dummy3 = DummyClassifier(strategy='prior', random_state = 100, constant = None)
dummy3.fit(x, y)
y_p3 = dummy3.predict(x)


print "     Dummy Model 3, strategy: prior, accuracy {0:.2f}, precision {0:.2f}, recall {0:.2f}, f1-score {0:.2f}"\
						.format(accuracy_score(y, y_p3), precision_score(y, y_p3), recall_score(y, y_p3), f1_score(y, y_p3))

dummy4 = DummyClassifier(strategy='uniform', random_state = 100, constant = None)
dummy4.fit(x, y)
y_p4 = dummy4.predict(x)


print "     Dummy Model 4, strategy: uniform, accuracy {0:.2f}, precision {0:.2f}, recall {0:.2f}, f1-score {0:.2f}"\
						.format(accuracy_score(y, y_p4), precision_score(y, y_p4), recall_score(y, y_p4), f1_score(y, y_p4))


















