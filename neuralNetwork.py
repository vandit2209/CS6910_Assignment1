from activations import ActiviationFunction
from loss import LossFunction
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
class NeuralNetwork:
    def __init__(self, arguments) -> None:
        self.args_backup = arguments # is saved for later variable purposes
        self.n_hidden_layers = arguments.num_layers
        self.neurons_h_layers = arguments.hidden_size
        self.h_activation_func_name = arguments.activation
        self.output_activation_func_name = arguments.output_activation
        self.weight_decay = arguments.weight_decay
        self.epsilon = arguments.epsilon
        self.learning_rate = arguments.learning_rate
        self.optimizer_name = arguments.optimizer
        self.loss_func_name = arguments.loss
        self.batch_size = arguments.batch_size
        self.epochs = arguments.epochs
        self.dataset_name = arguments.dataset

        self.load_dataset()

        self.layers = [self.x_train.shape[1]] + [self.neurons_h_layers]*self.n_hidden_layers + [self.y_train[0].shape[0]]
        self.n_layers = self.n_hidden_layers + 2

        self.activation = ActiviationFunction(arguments.activation)
        self.outputActivation = ActiviationFunction(arguments.output_activation)
        self.loss = LossFunction(arguments.loss)
        self.init_parameters()

    def load_dataset(self):
        preprocessor = StandardScaler()
        if self.dataset_name == "fashion_mnist":
            from keras.datasets import fashion_mnist
            (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()
        elif self.dataset_name == "mnist":
            from keras.datasets import mnist
            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        else:
            print("404 Dataset Not Found!")
            return

        self.x_train = self.x_train.astype('float64')
        self.y_train = self.y_train.astype('float64')
        self.x_test = self.x_test.astype('float64')
        self.y_test = self.y_test.astype('float64')

        self.x_train = self.x_train.reshape(self.x_train.shape[0],self.x_train.shape[1]*self.x_train.shape[2])
        self.x_train = preprocessor.fit_transform(self.x_train)
        self.y_train = self.y_train.reshape(self.y_train.shape[0],1)
        self.y_train = to_categorical(self.y_train)

        self.x_test = self.x_test.reshape(self.x_test.shape[0],self.x_test.shape[1]*self.x_test.shape[2])
        self.x_test = preprocessor.fit_transform(self.x_test)
        self.y_test = self.y_test.reshape(self.y_test.shape[0],1)
        self.y_test = to_categorical(self.y_test)

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.10, random_state=42)
    
    def init_parameters(self, debug = False):
        if self.args_backup.weight_init == "random":
            self.randomInit(debug)
        elif self.args_backup.weight_init == "xavier":
            self.xavierInit(debug)
    
    def randomInit(self, _print=False):
        self.parameters = {}
        constant = 0.04
        for i in range(1, self.n_layers):
            self.parameters[f"W{i}"] = np.random.randn(self.layers[i], self.layers[i-1])*constant
            self.parameters[f"b{i}"] = np.zeros((self.layers[i], 1))
            if(_print):
                print(f'W{i} -> {self.parameters["W" + str(i)].shape}')
                print(f'b{i} -> {self.parameters["b" + str(i)].shape}')
    
    def xavierInit(self, _print=False):
        self.parameters = {}
        for i in range(1, self.n_layers):
            self.parameters[f"W{i}"] = np.random.randn(self.layers[i], self.layers[i - 1]) * np.sqrt(2/ (self.layers[i - 1] + self.layers[i]))
            self.parameters[f"b{i}"] = np.zeros((self.layers[i], 1))
            if(_print):
                print(f'W{i} -> {self.parameters["W" + str(i)].shape}')
                print(f'b{i} -> {self.parameters["b" + str(i)].shape}')

    
    def gradsInit(self, debug = False):
        temp_gradients = {}
        for i in range(1, self.n_layers):
            temp_gradients[f"W{i}"] = np.zeros((self.layers[i], self.layers[i - 1]))
            temp_gradients[f"b{i}"] = np.zeros((self.layers[i], 1))
            if debug:
                print(f'W{i} -> {temp_gradients["W" + str(i)].shape}')
                print(f'b{i} -> {temp_gradients["b" + str(i)].shape}')
        return temp_gradients

    
    def forward_propagation(self, data):
        self.a = {}  # preactivation
        self.h = {}  # activation
        self.h["h0"] = data.reshape(len(data), 1)
        
        for i in range(1, self.n_layers - 1):
            self.a[f"a{i}"] = np.matmul(self.parameters[f"W{i}"].T,self.h[f"h{i-1}"]) + self.parameters[f"b{i}"]
            self.h[f"h{i}"] = self.activation.activate(self.a[f"a{i}"])
        
        # for output layer
        self.a[f"a{self.n_layers-1}"] = np.matmul(self.parameters[f"W{self.n_layers-1}"].T,self.h[f"h{self.n_layers-2}"]) + self.parameters[f"b{self.n_layers-1}"]
        self.h[f"h{self.n_layers - 1}"] = self.outputActivation.activate(self.a[f"a{i}"])
    

    def back_propagation(self, data):
        gradients = {}
        gradients[f"a{self.n_layers - 1}"] = self.loss.getGradient(data.T, self.h[f"h{self.n_layers - 1}"])
        for i in range(self.n_layers - 1, 1, -1):
            gradients[f"W{i}"] = np.dot(gradients[f'a{i}'], self.h[f'h{i-1}'])
            gradients[[f"b{i}"]] = gradients[f"a{i}"]
            gradients[f"h{i-1}"] = np.dot(self.parameters[f"W{i}"].T, gradients[f"a{i}"])
            gradients[f"a{i-1}"] = gradients[f"h{i-1}"] * self.activation.activate(self.a[f"a{i-1}"], backprop=True)
        
        gradients[f"W{1}"] = np.dot(gradients[f'a{1}'], self.h[f'h{1-1}'])
        gradients[[f"b{1}"]] = gradients[f"a{1}"]
        gradients[f"h{1-1}"] = np.dot(self.parameters[f"W{1}"].T, gradients[f"a{1}"])

        return gradients