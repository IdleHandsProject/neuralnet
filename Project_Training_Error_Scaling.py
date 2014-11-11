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

np.random.seed(0)

import time
from progressbar import ProgressBar


testGraphs = 0
pbar = ProgressBar()
pbar2 = ProgressBar()
totalpoint = 10000
testsize = totalpoint * 2/3
checksize = totalpoint * 1/3
print testsize
print checksize

training = Image.open("traindata.jpg").convert('L')
th = 100
edge = training.point(lambda i: i < th and 255)
edge = edge.filter(ImageFilter.FIND_EDGES)
edge = np.array(edge, dtype = 'uint8')
edge = (edge - 128)/128

print edge

Training = np.array(training, dtype = 'uint8')

print Training
Training = (Training -128)/128
OrigTrain = Training



Training = Training.flatten()

w = np.zeros(testsize, dtype=np.float)
l = np.zeros(testsize, dtype=np.float)
newtrain = np.zeros(testsize, dtype=np.int)
x = 0
lastx = 0
print OrigTrain.shape[0]
print OrigTrain.shape[1]

while x < testsize:
    w[x] = randint(11,OrigTrain.shape[0]-11)
    l[x] = randint(11,OrigTrain.shape[1]-11)
    
    if x % 2>0:
        if OrigTrain[w[x],l[x]] > 0:
            x = lastx
            print ("Not Black")
        else:
            if x < (testsize / 2):
                for box in range(-10,10):
                    if edge[w[x]+box, l[x]+box] < 1:
                        print ("Edge found")
                        newtrain[x] = 0
                        x += 1
                        break
            else:
                newtrain[x] = 0
                x += 1
                          
    else:
        if OrigTrain[w[x],l[x]] < 1:
            x = lastx
            print ("Not White")
            
        else:
            if x < (testsize / 2):
                for box in range(-10,10):
                    if edge[w[x]+box, l[x]+box] < 1:
                        print ("Edge found")
                        newtrain[x] = 1
                        x += 1
                        break
            else:
                newtrain[x] = 1
                x += 1
                
    print x
    lastx = x
w = w.reshape(testsize,1)
l = l.reshape(testsize,1)


coord = np.concatenate((l,w),axis=1)
print coord[:50:1]

##coord = coord.reshape(testsize,2)
##print coord 

 
##print newtrain[:testsize:1]
J = coord[:testsize:1] / 100
X = J
print X

y = newtrain[:testsize:1]



print ("Running Support Vector Machine")
print ("...")
 
print ("Testing prediction")
min_max_scaler = preprocessing.MinMaxScaler()

X = min_max_scaler.fit_transform(X)
##y = min_max_scaler.fit_transform(y)
fignum = 1
Y = y


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

	print X.shape
	print y.shape

	print X
	print y

	time.sleep(1)

	title = "Learning Curves (Naive Bayes)"
	# Cross validation with 100 iterations to get smoother mean test and train
	# score curves, each time with 20% data randomly selected as a validation set.
	print ("Still Working")
	time.sleep(3)
	cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=100,
									   test_size=0.2, random_state=0)

	estimator = GaussianNB()
	plot_learning_curve(estimator, title, X, y, ylim=(0.4, 0.6), cv=cv, n_jobs=4)
	print ("Still Working")
	time.sleep(3)
	title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
	# SVC is more expensive so we do a lower number of CV iterations:
	cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10,
									   test_size=0.2, random_state=0)
	estimator = SVC(gamma=0.001)
	plot_learning_curve(estimator, title, X, y, (0.4, 0.6), cv=cv, n_jobs=4)

	plt.show()

##################################TESTING###





# fit the model
for gamma in range(0,3):
    kernel = 'rbf'
    print ("Creating Linear Graph")
    clf = svm.SVC(kernel=kernel, gamma=gamma, verbose=2, cache_size=1000)
    clf.fit(X, y)

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(8, 8))
    plt.clf()

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=20, facecolors='none', zorder=10)
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)

    plt.axis('tight')
    x_min = 0
    x_max = 1   ##878/100 or 1
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
plt.savefig('gamma1.png')
plt.figure(2)
plt.savefig('gamma2.png')
plt.figure(3)
plt.savefig('gamma3.png')
plt.show()

ew = np.zeros(testsize, dtype=np.float)
el = np.zeros(testsize, dtype=np.float)

percent = 0
e = 0
print ("Checking Error")
for e in pbar(range(checksize)):
    ew = randint(11,OrigTrain.shape[0]-11)
    el = randint(11,OrigTrain.shape[1]-11)
    el /= 100
    ew /= 100
    check = clf.predict([[ew,el]])
    if check == OrigTrain[el,ew]:
        percent += 1
    
 
acc = float(percent) / checksize * 100
print ("The Percent Error is")
print acc 







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
time.sleep(2)

img = Image.fromarray(prediction, mode='L')
img.save("predictedimage.png")
img2 = Image.fromarray(OrigTrain, mode='L')
img2.save("OriginalTrain.png")
