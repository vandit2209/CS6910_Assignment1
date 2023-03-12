import numpy as np
class ActiviationFunction:
    def __init__(self, name):
        self.act_name = name
    
    @property
    def setname(self, new_name):
        self.act_name = new_name

    def activate(self, data, backprop = False):
        print(self.act_name)
        if self.act_name == "identity" and backprop:
            pass
        elif self.act_name == "identity" and not backprop:
            return self.identity(data)
        elif self.act_name == "sigmoid" and backprop:
            return self.backprop_sigmoid(data)
        elif self.act_name == "sigmoid" and not backprop:
            return self.sigmoid(data)
        elif self.act_name == "tanh" and backprop:
            return self.backprop_tanh(data)
        elif self.act_name == "tanh" and not backprop:
            return self.tanh(data)
        elif self.act_name == "ReLU" and backprop:
            pass
        elif self.act_name == "ReLU" and not backprop:
            return self.tanh(data)
        elif self.act_name == "softmax" and not backprop:
            return self.softmax(data)
        elif self.act_name == "softmax" and backprop:
            return self.backprop_softmax(data)
        

    
    def identity(self, data):
        return data
    
    def backprop_identity(self, data):
        pass

    def sigmoid(self, data):
        return 1/(1 + np.exp(-data))
    
    def backprop_sigmoid(self, data):
        temp = self.sigmoid(data)
        return (1 - temp)*temp
    
    def relu(self, data):
        return np.max(0, data)
    
    def backprop_relu(self, data):
        return np.heaviside(data, 1)
    
    def tanh(self, data):
        return np.tanh(data)
    
    def backprop_tanh(self, data):
        return 1 - np.square(np.tanh(data))
    
    def softmax(self, data):
        _max = np.max(data)
        numerator = np.exp(data - _max)
        denominator = np.sum(numerator)
        return numerator/denominator

    def backprop_softmax(self, data):
        _softmax = self.softmax(data)
        return _softmax * (1 - _softmax)
