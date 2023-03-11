import numpy as np
class Encoder:
    def __init__(self, debug = False):
        if debug:
            print("Encoder Initialized")
    def one_hot_encoder(self, data, classes):
        classes = list(classes)
        num_of_classes = len(classes)
        backup = [0]*num_of_classes
        encoded = []
        for elm in data:
            dump = backup.copy()
            dump[classes.index(elm)] = 1
            encoded.append(dump)
        
        return np.array(encoded)

