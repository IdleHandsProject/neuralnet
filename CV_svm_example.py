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
        params = dict( kernel_type = cv2.SVM_LINEAR, 
                       svm_type = cv2.SVM_C_SVC,
                       C = 1 )
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples])

import numpy as np
import cv2

samples = np.array(np.random.random((4,2)), dtype = np.float32)
y_train = np.array([1.,0.,0.,1.], dtype = np.float32)

clf = SVM()
clf.train(samples, y_train)
y_val = clf.predict(samples)

print y_val
