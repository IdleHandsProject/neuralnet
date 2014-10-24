import numpy as np
from numpy import indices
import matplotlib.pyplot as plt
import mlpy
from PIL import Image, ImageFilter, ImageEnhance
from scipy.misc import imresize
from sklearn.svm import SVC
from sklearn import svm
from random import randint
np.random.seed(0)
import time
from progressbar import ProgressBar

pbar = ProgressBar()

testsize = 20000

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


coord = np.indices(((Training.shape[0]),(Training.shape[1]))).swapaxes(0,2).swapaxes(0,1)
extra = coord

##coord = coord.reshape(1307*878,2)
##print coord

Training = Training.flatten()

w = np.zeros(testsize, dtype=np.int)
l = np.zeros(testsize, dtype=np.int)
newtrain = np.zeros(testsize, dtype=np.int)
x = 0
lastx = 0
while x < testsize:
    w[x] = randint(11,OrigTrain.shape[1]-11)
    l[x] = randint(11,OrigTrain.shape[0]-11)
    
    if x % 2>0:
        if OrigTrain[l[x],w[x]] > 0:
            x = lastx
            ##if x < 0:
            ##    x -= 1
            print ("Not Black")
        else:
            for box in range(-10,10):
                if edge[l[x]+box, w[x]+box] < 1:
                    print ("Edge found")
                    newtrain[x] = 1
                    x += 1
                    break          
    else:
        if OrigTrain[l[x],w[x]] < 1:
            x = lastx
            
            print ("Not White")
            
        else:
            for box in range(-10,10):
                if edge[l[x]+box, w[x]+box] < 1:
                    print ("Edge found")
                    newtrain[x] = 0
                    x += 1
                    break
    print x
    lastx = x
##print w
##print l

coord = np.concatenate((l,w),axis=0)
coord = coord.reshape(testsize,2)
##print coord 
 
##print newtrain[:testsize:1]
X = coord[:testsize:1]
y = newtrain[:testsize:1]
##y = Training[:60000:1]



clf = SVC()
print ("Running Support Vector Machine")
print ("...")
clf.fit(X, y) 
print ("Testing prediction")
print ("Point (100,200) should be 0")
print clf.predict([[100,200]])

prediction = np.zeros( (1307,878), dtype=np.int)
print ("Drawing Prediction Image - This may take a few minutes")
x = 0
for x in pbar(range(1307)):
    for z in range(0, 878):
	##checknum[x] = [randint(0,extra.shape[0]),randint(0,extra.shape[1])]
	##print checknum[x]
	prediction[x,z] = clf.predict([[x,z]]) * 255

print prediction
time.sleep(2)
##img = np.hstack((coord,prediction))
img = Image.fromarray(prediction, mode='L')
img.save("predictedimage.jpg")

fignum = 1
Y = y
# fit the model
#for kernel in ('linear','linear','linear'):
if fignum < 2:
    kernel = 'linear'
    print ("Creating Linear Graph")
    clf = svm.SVC(kernel=kernel, gamma=2)
    clf.fit(X, y)

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], 
s=20,
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
    fignum = fignum + 2
plt.show()
