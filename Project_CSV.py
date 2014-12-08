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
from sklearn import cross_validation
from numpy import genfromtxt

np.random.seed(0)

import time
from progressbar import ProgressBar

def epoch_range(start, end, step):
	while start <= end:
		yield start
		start += step

def box_range(start, end, step):
	while start <= end:
		yield start
		start += step

def swap_cols(arr, frm, to):
    arr[:,[frm, to]] = arr[:,[to, frm]]
		
		
epochs = 0
max_points = 15000
epoch_num = 1500
MSE = np.zeros((((max_points / epoch_num),1)))
MSEscore = np.zeros((((max_points / epoch_num),10)))
train_time = np.zeros((((max_points / epoch_num),1)))
####OPTIONS####

overSSH = 0
testGraphs = 0 
drawImage = 0
oldstuff = 0
gamma = 100
C = 100000


###############
print ("Gathering CSV Data")
image_data = genfromtxt('Data_List.csv', delimiter=',')
image_data = np.array(image_data, dtype= np.float)
swap_cols(image_data, 0, 1)
distance_data = image_data[:,[3]]
xy_data = image_data[:,:3] 
training_data = image_data[:,[2]]

	########################################### NEW DATA COLLECTION
for epochs in epoch_range(epoch_num, max_points, epoch_num):
	pbar = ProgressBar()
	pbar2 = ProgressBar()
	totalpoint =epochs
	
	testsize = totalpoint
	checksize = totalpoint * 1/3
	#print testsize
	#print checksize
	#testsize = 15000
		



	



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
	X = Training[:,:2]  / 1307
	y = Training[:,[2]].flatten()
	print X
	print y
	#min_max_scaler = preprocessing.MinMaxScaler()
	#y = min_max_scaler.fit_transform(y)
	fignum = 3
	Y = y
	
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.33, random_state=0)
	#training_start_time = time.time()
	clf = svm.SVC(C=C, kernel='rbf', gamma=gamma, verbose=1, cache_size=1000)
	#clf.fit(X_train, y_train)
	#training_time = time.time() - training_start_time
	#MSEscore[(epochs/epoch_num)-1] = clf.score(X_test, y_test)   
	#train_time[(epochs/epoch_num)-1] = training_time
	#time.sleep(1)
	#####################################################3
	training_start_time = time.time()
	scores = cross_validation.cross_val_score(clf, X, y, cv=10, scoring='accuracy')
	training_time = time.time() - training_start_time
	train_time[(epochs/epoch_num)-1] = training_time
	#print scores
	#time.sleep(10)
	MSEscore[(epochs/epoch_num)-1] = np.array(scores)
	#np.savetxt('MSE2.txt', MSE, delimiter=',')
np.savetxt('MSEscore.txt', MSEscore,delimiter=',')
np.savetxt('Timeepoch.txt', train_time,delimiter=',')
'''
	if oldstuff > 0:
		################################TESTING######################################
		if (testGraphs > 0):
			def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
									n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
				"""
				Generate a simple plot of the test and traning learning curve.

				Parameters
				----------
				estimator : object type that implements the "fit" and "predict" methods
					An object of that type which is cloned for each validation.

				title : string
					Title for the chart.

				X : array-like, shape (n_samples, n_features)
					
					Training vector, where n_samples is the number of samples and
					n_features is the number of features.

				y : array-like, shape (n_samples) or (n_samples, n_features), optional
					Target relative to X for classification or regression;
					None for unsupervised learning.

				ylim : tuple, shape (ymin, ymax), optional
					Defines minimum and maximum yvalues plotted.

				cv : integer, cross-validation generator, optional
					If an integer is passed, it is the number of folds (defaults to 3).
					Specific cross-validation objects can be passed, see
					sklearn.cross_validation module for the list of possible objects

				n_jobs : integer, optional
					Number of jobs to run in parallel (default 1).
				"""
				plt.figure()
				plt.title(title)
				if ylim is not None:
					plt.ylim(*ylim)
				plt.xlabel("Training examples")
				plt.ylabel("Score")
				train_sizes, train_scores, test_scores = learning_curve(
					estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
				train_scores_mean = np.mean(train_scores, axis=1)
				train_scores_std = np.std(train_scores, axis=1)
				test_scores_mean = np.mean(test_scores, axis=1)
				test_scores_std = np.std(test_scores, axis=1)
				plt.grid()

				plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
								 train_scores_mean + train_scores_std, alpha=0.1,
								 color="r")
				plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
								 test_scores_mean + test_scores_std, alpha=0.1, color="g")
				plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
						 label="Training score")
				plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
						 label="Cross-validation score")

				plt.legend(loc="best")
				return plt

			##from sklearn.datasets import load_digits
			##digits = load_digits()
			##X, y = digits.data, digits.target

			##y = digits.target[:666:1]

			#print X.shape
			#print y.shape

			#print X
			#print y

			

			#title = "Learning Curves (Naive Bayes)"
			# Cross validation with 100 iterations to get smoother mean test and train
			# score curves, each time with 20% data randomly selected as a validation set.
			#cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=100, test_size=0.2, random_state=0)

			#estimator = GaussianNB()
			#plot_learning_curve(estimator, title, X, y, ylim=(0, 1), cv=cv, n_jobs=4)
			title = "Learning Curves (SVM, RBF kernel, $\gamma=$,100)"
			# SVC is more expensive so we do a lower number of CV iterations:
			cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10,test_size=0.2, random_state=0)
			estimator = SVC(C=C,gamma=gamma,verbose=1)
			plot_learning_curve(estimator, title, X, y, (0.85, 1.05), cv=cv, n_jobs=4)
			if (overSSH > 0):
				print ("DONE!")

		##################################TESTING###

		
			
		kernel = 'rbf'
		print ("Creating SVM RBF")
		training_start_time = time.time()
		clf = svm.SVC(C=C, kernel=kernel, gamma=gamma, verbose=1, cache_size=1000)
		clf.fit(X, y)
		training_time = time.time() - training_start_time
		print (training_time)
		# fit the model
		if (overSSH > 0):
			for looping in range(0,1):
				kernel = 'rbf'
				print ("Creating Linear Graph")
				clf = svm.SVC(C=C, kernel=kernel, gamma=gamma, verbose=2, cache_size=1000)
				clf.fit(X, y)

				# plot the line, the points, and the nearest vectors to the plane
				plt.figure(fignum, figsize=(8, 8))
				plt.clf()

				plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=20, facecolors='none', zorder=10)
				plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)

				plt.axis('tight')
				x_min = 0
				x_max =  0.671  ##878/100 or 1
				y_min = 1   ##1307/100  or 1
				y_max = 0

				XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
				
				Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

				# Put the result into a color plot
				Z = Z.reshape(XX.shape)
				plt.figure(fignum, figsize=(8, 8))
				plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
				plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
							levels=[-.5, 0, .5])

				plt.xlim(x_min, x_max)
				plt.ylim(y_min, y_max)

				plt.xticks(())
				plt.yticks(())
				fignum += 1
		plt.figure(1)
		plt.savefig('LearningCurves.png')
		plt.figure(2)
		plt.savefig('Outline.png')
		#plt.figure(3)
		#plt.savefig('gamma3.png')
		#print clf.score(X,y)
		if (overSSH > 0):
			plt.show()

		ew = np.zeros(testsize, dtype=np.float)
		el = np.zeros(testsize, dtype=np.float)

		percent = 0
		e = 0
		check = np.zeros(checksize, dtype='int8')
		OriginalValue = np.zeros(checksize, dtype='int8')
		#print ("Checking Error")
		for e in pbar(range(checksize)):
			randcheck = randint(0,1147545)
			
			OriginalValue[e] = training_data[randcheck]
			
			#el /= 1307.00
			#ew /= 1307.00
			
			
			#ew = (((ew) * 1) / OrigTrain.shape[0])
			#el = (((el) * 1) / OrigTrain.shape[0])
			
			check[e] = clf.predict([image_data[randcheck,:2]/1307])
			
			if check[e] == OriginalValue[e]:
				percent += 1
				
			
		number = epochs / epoch_num
		#print OriginalValue
		#print check
		#time.sleep(5)
		MSE[number - 1] = mean_squared_error(OriginalValue, check) 
		#MSE[number - 1] = clf.score(X,y)
		#print MSE
		
		acc = float(percent) / checksize * 100
		print ("The Percent Accuracy is")
		print acc 
		time.sleep(1)





		if (drawImage > 0):
			testimagex = 1307
			testimagey = 878
			prediction = np.ones( (testimagex,testimagey), dtype=np.int)
			print ("Drawing Prediction Image - This may take a few minutes")
			x = 0
			for x in pbar2(range(testimagex)):
				for z in range(0, testimagey):
					xF = float(x)
					zF = float(z)
					prediction[x,z] = clf.predict([[(xF/100),(zF/100)]]) * 255


			print prediction.shape
			print OrigTrain.shape
			

			img = Image.fromarray(prediction, mode='L')
			img.save("predictedimage.png")
			img2 = Image.fromarray(OrigTrain, mode='L')
			img2.save("OriginalTrain.png")
print MSE		
np.savetxt('MSE.txt', MSE, delimiter=',')
'''