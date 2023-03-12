import wandb
import numpy as np
from tqdm.notebook import tqdm
from neuralNetwork import NeuralNetwork
class Optimiser(NeuralNetwork):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.parameters_without_activations = dict()
        self.separate_weights_and_biases()
    
    def update_parameters(self):
        for key in self.parameters_without_activations.keys():
            self.parameters[key] = self.parameters_without_activations[key]
    
    def separate_weights_and_biases(self):
        for key, value in self.parameters:
            if "W" in key or "b" in key:
                self.parameters_without_activations[key] = value
    
    def generateMetrics(self, x_data, y_data, epoch, _type = ""):
        predictions = []
        y = []
        yhat = []
        loss_history = []
        for x,yt in tqdm(zip(x_data , y_data), total=len(y_data), desc="Loss calculation", leave = False):
            self.h, self.a = self.forward_propagation(x)
            yhat = np.argmax(self.h['h' + str(self.L-1)])
            yt = yt.reshape(len(yt),1)
            _class = np.argmax(yt)
            y.append(_class)
            yhat.append(yhat)
            predictions.append(yhat == _class)
            loss_history.append(self.loss.getLoss(self.h['h' + str(self.L-1)],yt))

        accuracy = np.sum(predictions)/len(predictions)
        loss = np.sum(loss_history)/len(loss_history)

        print(f"{_type} Accuracy: {accuracy}", end = " ")
        print(f"{_type} Loss: {loss}", end = "\n")
        # wandb.log({f"{_type}_acc": accuracy, f"{_type}_loss": loss, "epoch": epoch})
        return yhat, y, accuracy*100, loss
    
    def stochastic_gradient_descent(self):
        _parameters = self.parameters_without_activations
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            gradients = self.gradsInit()
            for id, xy in tqdm(enumerate(zip(self.x_train, self.y_train)), desc = "Optimizer class = Stochastic Gradient Descent", leave = False):
                print(id, xy[0].shape, xy[1].shape)
                x = xy[0]
                y = xy[1]
                self.forward_propagation(x)
                delta = self.back_propagation(y)
                for key in gradients.keys():
                    gradients[key] = gradients[key] + delta[key]
                
                if (id + 1) % self.batch_size == 0:
                    for key in _parameters.keys():
                        _parameters[key] = _parameters[key] - self.learning_rate * gradients[key]
                        gradients = self.gradsInit()
        
        
            self.generateMetrics(self.x_val, self.y_val, epoch, _type = "Validation")
            self.generateMetrics(self.x_val, self.y_val,epoch, _type = "Train")
            self.generateMetrics(self.x_test, self.y_test, epoch,_type = "Test")

        self.update_parameters()
    
    def moment_based_gradient_descent(self):
        self.momentum = self.args_backup.momentum
        _parameters = self.parameters_without_activations
        _global = self.gradsInit()
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            gradients = self.gradsInit()
            lookahead = self.gradsInit()
            for id, x, y in tqdm(enumerate(zip(self.x_train, self.y_train)), desc = "Optimizer class = Momentum Based Gradient Descent", leave = False):
                self.forward_propagation(x)
                delta = self.back_propagation(y)
                for key in gradients.keys():
                    gradients[key] += delta[key]
                
                if (id + 1) % self.batch_size == 0:
                    for key in lookahead.keys():
                        lookahead[key] = self.momentum * _global[key] + self.learning_rate * gradients[key]
                    
                        _parameters[key] = _parameters[key] - lookahead[key]
                    
                        _global[key] = lookahead[key]
                    
                    gradients = self.gradsInit()
        
            self.generateMetrics(self.x_val, self.y_val, epoch, _type = "Validation")
            self.generateMetrics(self.x_val, self.y_val,epoch, _type = "Train")
            self.generateMetrics(self.x_test, self.y_test, epoch,_type = "Test")
            
        self.update_parameters()
    
    def nestrov_gradient_descent(self):
        self.momentum = self.args_backup.momentum # beta
        _parameters = self.parameters_without_activations
        _global = self.gradsInit()
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            gradients = self.gradsInit()
            lookahead = self.gradsInit()

            # partial init
            for key in lookahead.keys():
                    lookahead[key] = self.momentum * lookahead[key]
            
            for id, x, y in tqdm(enumerate(zip(self.x_train, self.y_train)), desc = "Optimizer class = Nestrov Gradient Descent", leave = False):
                self.forward_propagation(x)
                delta = self.back_propagation(y)
                for key in gradients.keys():
                    gradients[key] += delta[key]

                
                if (id + 1) % self.batch_size == 0:
                    for key in lookahead.keys():
                        lookahead[key] = self.momentum * lookahead[key] + self.learning_rate * gradients[key]
                    
                        _parameters[key] = _parameters[key] - lookahead[key]
                    
                        _global[key] = lookahead[key]

                        gradients = self.gradsInit()

            self.generateMetrics(self.x_val, self.y_val, epoch, _type = "Validation")
            self.generateMetrics(self.x_val, self.y_val,epoch, _type = "Train")
            self.generateMetrics(self.x_test, self.y_test, epoch,_type = "Test")

        self.update_parameters()
    
    def rms_prop(self):
        self.beta = self.args_backup.beta
        self.epsilon = self.args_backup.epsilon
        _parameters = self.parameters_without_activations
        _global = self.gradsInit()
        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            gradients = self.gradsInit()
            for id, x, y in tqdm(enumerate(zip(self.x_train, self.y_train)), desc = "Optimizer class = RMS Prop", leave = False):
                self.forward_propagation(x)
                delta = self.back_propagation(y)
                for key in gradients.keys():
                    gradients[key] += delta[key]

                
                if (id + 1) % self.batch_size == 0:
                    for key in _global.keys():
                        _global[key] = self.beta * _global[key] + (1 - self.beta) * (gradients[key]**2)
                    
                        _parameters[key] = _parameters[key] - (self.learning_rate * gradients[key])/ (np.sqrt(_global[key] + self.epsilon))
                    
                    gradients = self.gradsInit()
        

            self.generateMetrics(self.x_val, self.y_val, epoch, _type = "Validation")
            self.generateMetrics(self.x_val, self.y_val,epoch, _type = "Train")
            self.generateMetrics(self.x_test, self.y_test, epoch,_type = "Test")

        self.update_parameters()
    

    def adam(self):
        self.beta1 = self.args_backup.beta1
        self.beta2 = self.args_backup.beta2
        self.epsilon = self.args_backup.epsilon

        _parameters = self.parameters_without_activations

        m_gradients = self.gradsInit()
        v_gradients = self.gradsInit()

        m_gradients_hat = self.gradsInit()
        v_gradients_hat = self.gradsInit()

        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            gradients = self.gradsInit()
            for id, x, y in tqdm(enumerate(zip(self.x_train, self.y_train)), desc = "Optimizer class = Adam", leave = False):
                self.forward_propagation(x)
                delta = self.back_propagation(y)
                for key in gradients.keys():
                    gradients[key] += delta[key]
                

                if (id+1) % self.batch_size == 0:
                    for key in m_gradients.keys():
                        m_gradients[key] = self.beta1 * m_gradients[key] + (1 - self.beta1) * gradients[key]
                    
                        v_gradients[key] = self.beta2 * v_gradients[key] + (1 - self.beta2) * (gradients[key]**2)

                    
                        m_gradients_hat[key] = m_gradients[key] / (1 - self.beta1 ** (epoch + 1))
                        v_gradients_hat[key] = v_gradients[key] / (1 - self.beta2 ** (epoch + 1))

                    
                    for key in _parameters.keys():
                        _parameters[key] = _parameters[key] - (self.learning_rate * m_gradients_hat[key]) / np.sqrt(v_gradients_hat[key] + self.epsilon)
                    
                    gradients = self.gradsInit()
                
            self.generateMetrics(self.x_val, self.y_val, epoch, _type = "Validation")
            self.generateMetrics(self.x_val, self.y_val,epoch, _type = "Train")
            self.generateMetrics(self.x_test, self.y_test, epoch,_type = "Test")

        self.update_parameters()
    

    def nadam(self):
        self.beta = self.args_backup.momentum
        self.beta1 = self.args_backup.beta1
        self.beta2 = self.args_backup.beta2
        self.epsilon = self.args_backup.epsilon

        _parameters = self.parameters_without_activations
        _global = self.gradsInit()

        m_gradients = self.gradsInit()
        v_gradients = self.gradsInit()

        m_gradients_hat = self.gradsInit()
        v_gradients_hat = self.gradsInit()

        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            gradients = self.gradsInit()
            lookahead = self.gradsInit()

            # partial init
            for key in lookahead.keys():
                lookahead[key] = self.beta * lookahead[key]

            for id, x, y in tqdm(enumerate(zip(self.x_train, self.y_train)), desc = "Optimizer class = nAdam", leave = False):
                self.forward_propagation(x)
                delta = self.back_propagation(y)
                for key in gradients.keys():
                    gradients[key] += delta[key]
                

                if (id+1) % self.batch_size == 0:
                    for key in lookahead.keys():
                        lookahead[key] = self.beta * _global[key] + self.learning_rate * gradients[key]
                        m_gradients[key] = self.beta1 * m_gradients[key] + (1 - self.beta1) * gradients[key]
                    
                        v_gradients[key] = self.beta2 * v_gradients[key] + (1 - self.beta2) * (gradients[key]**2)
                    
                        m_gradients_hat[key] = m_gradients[key] / (1 - self.beta1 ** (epoch + 1))
                        v_gradients_hat[key] = v_gradients[key] / (1 - self.beta2 ** (epoch + 1))

                        _parameters[key] = _parameters[key] - (self.learning_rate * m_gradients_hat[key]) / np.sqrt(v_gradients_hat + self.epsilon)
                    
                    for key in _global.keys():
                        _global[key] = lookahead[key]

                    
                    gradients = self.gradsInit()
            
            self.generateMetrics(self.x_val, self.y_val, epoch, _type = "Validation")
            self.generateMetrics(self.x_val, self.y_val,epoch, _type = "Train")
            self.generateMetrics(self.x_test, self.y_test, epoch,_type = "Test")

        self.update_parameters()

    def train(self):
        if self.optimizer_name == "sgd":
            self.stochastic_gradient_descent()
        elif self.optimizer_name == "momentum":
            self.moment_based_gradient_descent()
        elif self.optimizer_name == "nag":
            self.nestrov_gradient_descent()
        elif self.optimizer_name == "rmsprop":
            self.rms_prop()
        elif self.optimizer_name == "adam":
            self.adam()
        elif self.optimizer_name == "nadam":
            self.nadam()
        else:
            raise Exception('The choice of optimizer is not found. Please select from ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]')
    