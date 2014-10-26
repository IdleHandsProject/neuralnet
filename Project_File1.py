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
pbar2 = ProgressBar()
totalpoint = 30000
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


##coord = np.indices(((Training.shape[0]),(Training.shape[1]))).swapaxes(0,2).swapaxes(0,1)
##extra = coord
##print coord


Training = Training.flatten()

w = np.zeros(testsize, dtype=np.float)
l = np.zeros(testsize, dtype=np.float)
newtrain = np.zeros(testsize, dtype=np.int)
x = 0
lastx = 0
print OrigTrain.shape[1]
print OrigTrain.shape[0]

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

fignum = 1
Y = y
# fit the model
for gamma in range(0,4):
    kernel = 'rbf'
    print ("Creating Linear Graph")
    clf = svm.SVC(kernel=kernel, gamma=gamma, verbose=2, tol=0.1)
    clf.fit(X, y)

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(8, 8))
    plt.clf()

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=20, facecolors='none', zorder=10)
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)

    plt.axis('tight')
    x_min = 0
    x_max = 900/100
    y_min = 1308/100
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
plt.show()

ew = np.zeros(testsize, dtype=np.float)
el = np.zeros(testsize, dtype=np.float)

percent = 0
print ("Checking Error")
for e in pbar(range(checksize)):
    ew = randint(11,OrigTrain.shape[0]-11)
    el = randint(11,OrigTrain.shape[1]-11)
    el /= 100
    ew /= 100
    check = clf.predict([[ew,el]])
    if check == OrigTrain[ew,el]:
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
        prediction[x,z] = clf.predict([[xF,zF]]) * 255


print prediction
time.sleep(2)

img = Image.fromarray(prediction, mode='L')
img.save("predictedimage.png")

