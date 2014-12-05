import numpy as np
from numpy import indices
import matplotlib.pyplot as plt
import mlpy
from PIL import Image, ImageFilter, ImageEnhance
from scipy.misc import imresize
from sklearn.svm import SVC
from sklearn import svm
from random import randint
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.learning_curve import learning_curve
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


np.random.seed(0)

import time
from progressbar import ProgressBar

def epoch_range(start, end, step):
	while start <= end:
		yield start
		start += step


epochs = 0
max_points = 10000
epoch_num = 1000
MSE = np.zeros((((max_points / epoch_num),1)))

####OPTIONS####

overSSH = 0
testGraphs = 0
drawImage = 0
gamma = 500
C = 100000

###############

######## Training Data Processing########

pbar = ProgressBar()
pbar2 = ProgressBar()
totalpoint = 10000
testsize = totalpoint * 2/3
checksize = totalpoint * 1/3
#print testsize
#print checksize

training = Image.open("TrainingData.jpg").convert('L')
th = 100
edge = training.point(lambda i: i < th and 255)
edge = edge.filter(ImageFilter.FIND_EDGES)
edge = np.array(edge, dtype = 'uint8')
edge = (edge - 128)/128

#print edge

Training = np.array(training, dtype = 'uint8')



#print Training
Training = (Training -128)/128
OrigTrain = Training



Training = Training.flatten()
#print Training 

w = np.zeros(testsize, dtype=np.float)
l = np.zeros(testsize, dtype=np.float)
newtrain = np.zeros(testsize, dtype=np.int)
x = 0
lastx = 0
#print OrigTrain.shape[0]
#print OrigTrain.shape[1]
boxsize = 10

while x < testsize:
	w[x] = randint(11,OrigTrain.shape[0]-11)
	l[x] = randint(11,OrigTrain.shape[1]-11)
	
	if x % 2>0:
		if OrigTrain[w[x],l[x]] > 0:
			x = lastx
			#print ("Not Black")
		else:
			if x < (testsize / 2):
				for box in range((-1 * boxsize),boxsize):
					if edge[w[x]+box, l[x]+box] < 1:
						#print ("Edge found")
						newtrain[x] = 0
						x += 1
						break
			else:
				newtrain[x] = 0
				x += 1
						  
	else:
		if OrigTrain[w[x],l[x]] < 1:
			x = lastx
			#print ("Not White")
			
		else:
			if x < (testsize / 2):
				for box in range((-1 * boxsize),boxsize):
					if edge[w[x]+box, l[x]+box] < 1:
						#print ("Edge found")
						newtrain[x] = 1
						x += 1
						break
			else:
				newtrain[x] = 1
				x += 1
				
	#print x
	lastx = x
w = w.reshape(testsize,1)
l = l.reshape(testsize,1)


coord = np.concatenate((l,w),axis=1)
#print coord[:50:1]
##coord = coord.reshape(testsize,2)
##print coord 

 
##print newtrain[:testsize:1]
J = coord[:testsize:1] / 1300
X = J

y = newtrain[:testsize:1]
#print y 
	


#print ("Running Support Vector Machine")
#print ("...")
 
#print ("Testing prediction")
min_max_scaler = preprocessing.MinMaxScaler()
#X = min_max_scaler.fit_transform(X)
##y = min_max_scaler.fit_transform(y)
fignum = 3
Y = y
	
	######### Data Processing End ##########
	
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [10000,100000,200000,300000], 'gamma': [500,400,300,200,100], 'kernel': ['rbf']},
 ]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)


	
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [500,400,300,200,100],
                     'C': [10000,100000,200000,300000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
	
