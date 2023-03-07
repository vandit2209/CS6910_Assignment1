# %%
import wandb
from keras.datasets import fashion_mnist
import numpy as np
import cli_args
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

# %%
# wandb.init (
# # set the wand project where this run will be logged
# project="CS6910_Assignment_1",
# )
# # Loading the fashion-MNIST dataset
# (x_train, y_train), (_test, _test) = fashion_mnist.load_data ()
# #class names for fashion-MIST
# class_names = ['T-shirt',
# 'Trouser',
# 'Pullover',
# 'Dress',
# 'Sandal', 'Shirt',
# 'Sneaker',
# 'Bag',
# 'Coat',
# 'Ankle boot']
# # creating 2x5 grid
# img={}
# for i in range(10):
#     # to find first image in the training set with class label i
#     idx = np.where (y_train == i)[0][0]
#     # Plot the image
#     img[class_names[i]] = wandb.Image(x_train[idx], caption=class_names[i])
# wandb.log(img)
# # [optional] finish the wand run, necessary in notebooks
# wandb.finish()

# %%
arguments = cli_args.argumentsIntake()

# %%
class ActiviationFunction:
    def __init__(self, name):
        self.act_name = name
    
    @property
    def setname(self, new_name):
        self.act_name = new_name

    def activate(self, data, backprop = False):
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

    
    def identity(self, data):
        return data

    def sigmoid(self, data):
        return 1/(1 + np.exp(-data))
    
    def backprop_sigmoid(self, data):
        temp = self.sigmoid(data)
        return (1 - temp)*temp
    
    def relu(self, data):
        return np.max(0, data)
    
    def tanh(self, data):
        # return (np.exp(data) - np.exp(-data))/(np.exp(data) + np.exp(-data))
        return np.tanh(data)
    
    def backprop_tanh(self, data):
        return 1 - np.square(np.tanh(data))
    



# %%
class LossFunction:
    def __init__(self, name):
        self.name = name
    
    def getLoss(self, data):
        if self.name == "cross_entropy":
            pass
        elif self.name == "squared_error":
            pass

# %%
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

        self.layers = [self.x_train.shape[1]] + self.neurons_h_layers + [self.y_train[0].shape[0]]
        self.n_layers = self.n_hidden_layers + 2

        self.activation = ActiviationFunction(arguments.activation)
        self.outputActivation = ActiviationFunction(arguments.output_activation)

        self.init_parameters()
    
    def load_dataset(self):
        preprocessor = StandardScaler()
        if self.dataset_name == "fashion_mnist":
            from keras.datasets import fashion_mnist
            (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()

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
            
        else:
            pass
    
    def init_parameters(self):
        if self.args_backup.weight_init == "random":
            self.randomInit(_print = True)
        elif self.args_backup.weight_init == "xavier":
            self.xavierInit(_print = True)
    
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
            self.parameters[f"W{i}"] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i - 1]) * np.sqrt(2/ (self.layer_sizes[i - 1] + self.layer_sizes[i]))
            self.parameters[f"b{i}"] = np.zeros((self.layers[i], 1))
            if(_print):
                print(f'W{i} -> {self.parameters["W" + str(i)].shape}')
                print(f'b{i} -> {self.parameters["b" + str(i)].shape}')

    
    def forward_propagation(self, data):
        self.a = {}  # preactivation
        self.h = {}  # activation
        self.h["h0"] = data.reshape(len(data),1)
        
        for i in range(1, self.n_layers - 1):
            print(f"W{self.parameters[f'W{i}'].shape} x h{self.h[f'h{i-1}'].shape} + b{self.parameters[f'b{i}'].shape}")
            self.a[f"a{i}"] = np.matmul(self.parameters[f"W{i}"],self.h[f"h{i-1}"]) + self.parameters[f"b{i}"]
            self.h[f"h{i}"] = self.activation.activate(self.a[f"a{i}"])
        
        # for output layer
        self.a[f"a{self.n_layers-1}"] = np.matmul(self.parameters[f"W{self.n_layers-1}"],self.h[f"h{self.n_layers-2}"]) + self.parameters[f"b{self.n_layers-1}"]
        self.h[f"h{self.n_layers - 1}"] = self.outputActivation.activate(self.a[f"a{i}"])
        print(self.h["h0"].shape)
        print(self.h["h1"].shape)

    def back_propagation(self, data):
        pass


# %%

x = NeuralNetwork(arguments)
x.forward_propagation(x.x_train[0])




# %%
