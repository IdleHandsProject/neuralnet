from PIL import Image
from numpy import array
import numpy as np
import cv2

class StatModel(object):
    '''parent class - starting point to add abstraction'''    
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''
    def __init__(self):
        self.model = cv2.SVM()
    def train(self, samples, responses):
        #setting algorithm parameters
        params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C = 1 )
        self.model.train(samples, responses, params = params)
    def predict(self, samples):
        return np.uint8( [self.model.predict(s) for s in samples])
		
test_image = Image.open("OriginalDesat.jpg").convert('L')
test_array = array(test_image, dtype = np.uint8);

train_image = Image.open("TrainingDataBW.jpg").convert('L')
train_array = array(train_image, dtype = np.uint8)

clf = SVM()
clf.train(test_array, train_array)
y_val = clf.predict(samples)

