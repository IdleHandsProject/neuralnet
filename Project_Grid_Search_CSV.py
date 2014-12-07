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
from numpy import genfromtxt

np.random.seed(0)

import time
from progressbar import ProgressBar

def epoch_range(start, end, step):
	while start <= end:
		yield start
		start += step

def swap_cols(arr, frm, to):
    arr[:,[frm, to]] = arr[:,[to, frm]]
		

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

print ("Gathering CSV Data")
image_data = genfromtxt('Data_List.csv', delimiter=',')
image_data = np.array(image_data, dtype= np.float)
swap_cols(image_data, 0, 1)
distance_data = image_data[:,[3]]
xy_data = image_data[:,:3] 
training_data = image_data[:,[2]]

######## Training Data Processing########

pbar = ProgressBar()
pbar2 = ProgressBar()
totalpoint = 10000
testsize = totalpoint * 2/3
checksize = totalpoint * 1/3
#print testsize
#print checksize

Training = np.zeros([testsize,3], dtype=np.float)
last_range = 0
point_num = 0
point_find = 0
lastpoint_find = 0 
set_distance = [0,5,25,50,75,100]
set_separation = [5,20,25,25,25,185] 
set_size =  [testsize * 0.167,testsize*0.30,testsize*0.233,testsize*0.133,testsize*0.100,testsize*0.067]
set_size = np.array(set_size)
print set_size
for set in range(0,set_size.size):
	point_num += set_size[set]
	print "finding points %s to %s" %(last_range, point_num)
	print "number of points %s" %(point_num - last_range)
	
	if point_num > testsize:
		point_num=testsize
	while point_find < point_num:
	#for point_find in range(last_range,point_num):
		print "point_find %s" %point_find
		point = randint(0,1147545)
		if (set_distance[set] < distance_data[point]) and ( distance_data[point]< set_distance[set]+set_separation[set]) :
			if point_find % 2 > 0:
				if  training_data[point] > 0:
					point_find = lastpoint_find
					print ("Not Black")
				else:
					print xy_data[point]
					Training[point_find] = xy_data[point]
					point_find += 1
					#time.sleep(1)
					
					
			else:
				if  training_data[point] < 0:
					point_find = lastpoint_find
					print ("Not White")
				else:
					print xy_data[point]
					Training[point_find] = xy_data[point]
					point_find += 1
					#time.sleep(1)
			
		lastpoint_find = point_find
		
	last_range = point_num
		
			
	
time.sleep(1)

X = Training[:(Training.shape[0]*3/3),:2]  / 1307

y = Training[:(Training.shape[0]*3/3),[2]].flatten()
print X
print y
#min_max_scaler = preprocessing.MinMaxScaler()
#y = min_max_scaler.fit_transform(y)
fignum = 3
Y = y

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
  {'C': [10000,100000,200000,300000], 'gamma': [100,50,25,1,0.1], 'kernel': ['rbf']},
 ]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)


	
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [100,50,25,1,0.1],
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
	
