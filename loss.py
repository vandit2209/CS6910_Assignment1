import numpy as np
class LossFunction:
    def __init__(self, name):
        self.name = name
    
    def getLoss(self, y, yhat):
        if self.name == "cross_entropy":
            return self.cross_entropy_loss(y, yhat)
        elif self.name == "mean_squared_error":
            return self.mean_squared_error_loss(y, yhat)
    
    def getGradient(self, y, yhat):
        if self.name == "cross_entropy":
            return self.backprop_cross_entropy(y, yhat)
        elif self.name == "mean_squared_error":
            return self.backprop_mean_squared_loss(y, yhat)
    
    def cross_entropy_loss(self, y, yhat):
        probability_predicted = yhat[np.argmax(y)]
        if probability_predicted <= 0:
            probability_predicted += 10**(-9)
        return -np.log(probability_predicted)
    
    def mean_squared_error_loss(self,y, yhat):
        return (1/2)*np.sum(np.square(yhat - y))
    
    def backprop_cross_entropy(self, y, yhat):
        return yhat - y
    
    def backprop_mean_squared_loss(self, y, yhat):
        return (yhat - y) * yhat * (1 - yhat)