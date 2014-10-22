import numpy as np
from numpy import indices
import matplotlib.pyplot as plt
import mlpy
from PIL import Image
from scipy.misc import imresize
from sklearn.svm import SVC
from sklearn import svm
from random import randint
np.random.seed(0)


img = Image.open("OriginalDesat.jpg").convert('L')
training = Image.open("TrainingDataBW.jpg").convert('L')
f = np.array(img, dtype = 'float32')
Training = np.array(training, dtype = 'uint8')

Training = (Training -128)/128
OrigTrain = Training
##f = f - 128
##f = f / 128
f = np.indices(((Training.shape[0]),(Training.shape[1]))).swapaxes(0,2).swapaxes(0,1)
extra = f
##f = np.random.rand(10,2)
##f = (f*2)-1

f = f.reshape(1307*878,2)
##f = f[:f.shape[0],:f.shape[1]]
#f = Training[:,:]
Training = Training.flatten()
grid = [0,0]

for x in range(0, 30000):
    w = randint(0,OrigTrain.shape[1])
    l = randint(0,OrigTrain.shape[0])
    grid[x] = (w,l)
    color[x] = 
    if (x % 2) > 0:
	if color[x] > 0:
	    x = x - 1
print grid



print f[:500000:100]
print Training[:500000:100]
X = f[:500000:5000]
y = Training[:500000:5000]
#X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
#y = np.array([1, 1, 2, 2])


clf = SVC()
clf.fit(X, y) 
print randint(0,extra.shape[0])
for x in range (0, 50):
	##checknum[x] = [randint(0,extra.shape[0]),randint(0,extra.shape[1])]
	##print checknum[x]
	print (clf.predict([[randint(0,extra.shape[0]),randint(0,extra.shape[1])]]))

fignum = 1
Y = y
# fit the model
for kernel in ('linear','linear','linear'):
    print ("test")
    clf = svm.SVC(kernel=kernel, gamma=2)
    clf.fit(X, Y)

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10)
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)

    plt.axis('tight')
    x_min = -1000
    x_max = 1000
    y_min = -1000
    y_max = 1000

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1
plt.show()
