class Optimiser(NeuralNetwork):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.parameters_without_activations = None
        self.separate_weights_and_biases()
    
    def separate_weights_and_biases(self):
        for key, value in self.parameters:
            if "W" in key or "B" in key:
                self.parameters_without_activations[key] = value
    
    def generateMetrics(self, x_data, y_data):
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
			
        return yhat, y, accuracy*100, loss
    
    def stochastic_gradient_descent(self):
        _parameters = self.parameters_without_activations
        for i in tqdm(range(self.epochs), desc="Epochs"):
            gradients = self.gradsInit()
            for id, x, y in tqdm(enumerate(zip(self.x_train, self.y_train)), desc = "Optimizer class = Stochastic Gradient Descent", leave = False):
                self.forward_propagation(x)
                delta = self.back_propagation(y)
                for key in gradients.keys():
                    gradients[key] = gradients[key] + delta[key]
                
                if (id + 1) % self.batch_size == 0:
                    for key in _parameters.keys():
                        _parameters[key] = _parameters[key] - self.learning_rate * gradients[key]
                        gradients = self.gradsInit()
        
        validation_yhat, validation_y, validation_acc, validation_loss = self.generateMetrics(self.x_val, self.y_val)
        print(f"Validation Accuracy: {validation_acc}", end = " ")
        print(f"Validation Loss: {validation_loss}", end = "\n")


        train_yhat, train_y, train_acc, train_loss = self.generateMetrics(self.x_val, self.y_val)
        print(f"Train Accuracy: {train_acc}", end = "")
        print(f"Train Loss: {train_loss}", end = "\n")

        test_yhat, test_y, test_acc, test_loss = self.generateMetrics(self.x_test, self.y_test)
        print(f"Test Accuracy: {test_acc}", end = " ")
        print(f"Test Loss: {test_loss}", end = "\n")
            
        return _parameters
    
    def moment_based_gradient_descent(self):
        self.momentum = self.args_backup.momentum
        _parameters = self.parameters_without_activations
        _global = self.gradsInit()
        for i in tqdm(range(self.epochs), desc="Epochs"):
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
                    
                    for key in _parameters.keys():
                        _parameters[key] = _parameters[key] - lookahead[key]
                    
                    for key in _global.keys():
                        _global[key] = lookahead[key]
                    
                    gradients = self.gradsInit()
        
        validation_yhat, validation_y, validation_acc, validation_loss = self.generateMetrics(self.x_val, self.y_val)
        print(f"Validation Accuracy: {validation_acc}", end = " ")
        print(f"Validation Loss: {validation_loss}", end = "\n")


        train_yhat, train_y, train_acc, train_loss = self.generateMetrics(self.x_val, self.y_val)
        print(f"Train Accuracy: {train_acc}", end = "")
        print(f"Train Loss: {train_loss}", end = "\n")

        test_yhat, test_y, test_acc, test_loss = self.generateMetrics(self.x_test, self.y_test)
        print(f"Test Accuracy: {test_acc}", end = " ")
        print(f"Test Loss: {test_loss}", end = "\n")
            
        return _parameters
    
    def nestrov_gradient_descent(self):
        self.momentum = self.args_backup.momentum # beta
        _parameters = self.parameters_without_activations
        _global = self.gradsInit()
        for i in tqdm(range(self.epochs), desc="Epochs"):
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
                    
                    for key in _parameters.keys():
                        _parameters[key] = _parameters[key] - lookahead[key]
                    
                    for key in _global.keys():
                        _global[key] = lookahead[key]

                        gradients = self.gradsInit()

        validation_yhat, validation_y, validation_acc, validation_loss = self.generateMetrics(self.x_val, self.y_val)
        print(f"Validation Accuracy: {validation_acc}", end = " ")
        print(f"Validation Loss: {validation_loss}", end = "\n")


        train_yhat, train_y, train_acc, train_loss = self.generateMetrics(self.x_val, self.y_val)
        print(f"Train Accuracy: {train_acc}", end = "")
        print(f"Train Loss: {train_loss}", end = "\n")

        test_yhat, test_y, test_acc, test_loss = self.generateMetrics(self.x_test, self.y_test)
        print(f"Test Accuracy: {test_acc}", end = " ")
        print(f"Test Loss: {test_loss}", end = "\n")

        return _parameters
    
    def rms_prop(self):
        self.beta = self.args_backup.beta
        self.epsilon = self.args_backup.epsilon
        _parameters = self.parameters_without_activations
        _global = self.gradsInit()
        for i in tqdm(range(self.epochs), desc="Epochs"):
            gradients = self.gradsInit()
            for id, x, y in tqdm(enumerate(zip(self.x_train, self.y_train)), desc = "Optimizer class = RMS Prop", leave = False):
                self.forward_propagation(x)
                delta = self.back_propagation(y)
                for key in gradients.keys():
                    gradients[key] += delta[key]

                
                if (id + 1) % self.batch_size == 0:
                    for key in _global.keys():
                        _global[key] = self.beta * _global[key] + (1 - self.beta) * (gradients[key]**2)
                    
                    for key in _parameters.keys():
                        _parameters[key] = _parameters[key] - (self.learning_rate * gradients[key])/ (np.sqrt(_global[key] + self.epsilon))
                    
                    gradients = self.gradsInit()
        

        validation_yhat, validation_y, validation_acc, validation_loss = self.generateMetrics(self.x_val, self.y_val)
        print(f"Validation Accuracy: {validation_acc}", end = " ")
        print(f"Validation Loss: {validation_loss}", end = "\n")


        train_yhat, train_y, train_acc, train_loss = self.generateMetrics(self.x_val, self.y_val)
        print(f"Train Accuracy: {train_acc}", end = "")
        print(f"Train Loss: {train_loss}", end = "\n")

        test_yhat, test_y, test_acc, test_loss = self.generateMetrics(self.x_test, self.y_test)
        print(f"Test Accuracy: {test_acc}", end = " ")
        print(f"Test Loss: {test_loss}", end = "\n")

        return  _parameters
    

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
            for id, x, y in tqdm(enumerate(zip(self.x_train, self.y_train)), desc = "Optimizer class = RMS Prop", leave = False):
                self.forward_propagation(x)
                delta = self.back_propagation(y)
                for key in gradients.keys():
                    gradients[key] += delta[key]
                

                if (id+1) % self.batch_size == 0:
                    for key in m_gradients.keys():
                        m_gradients[key] = self.beta1 * m_gradients[key] + (1 - self.beta1) * gradients[key]
                    
                    for key in v_gradients.keys():
                        v_gradients[key] = self.beta2 * v_gradients[key] + (1 - self.beta2) * (gradients[key]**2)

                    
                    for key in _parameters.keys():
                        m_gradients_hat[key] = m_gradients[key] / (1 - self.beta1 ** (epoch + 1))
                        v_gradients_hat[key] = v_gradients[key] / (1 - self.beta2 ** (epoch + 1))

                    
                    for key in _parameters.keys():
                        _parameters[key] = _parameters[key] - (self.learning_rate * m_gradients_hat[key]) / np.sqrt(v_gradients_hat + self.epsilon)
                    
                    gradients = self.gradsInit()
                
        validation_yhat, validation_y, validation_acc, validation_loss = self.generateMetrics(self.x_val, self.y_val)
        print(f"Validation Accuracy: {validation_acc}", end = " ")
        print(f"Validation Loss: {validation_loss}", end = "\n")


        train_yhat, train_y, train_acc, train_loss = self.generateMetrics(self.x_val, self.y_val)
        print(f"Train Accuracy: {train_acc}", end = "")
        print(f"Train Loss: {train_loss}", end = "\n")

        test_yhat, test_y, test_acc, test_loss = self.generateMetrics(self.x_test, self.y_test)
        print(f"Test Accuracy: {test_acc}", end = " ")
        print(f"Test Loss: {test_loss}", end = "\n")

        return  _parameters
    

    def nadam(self):
        self.beta = self.args_backup.beta # don't know what to put here yet
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

            for id, x, y in tqdm(enumerate(zip(self.x_train, self.y_train)), desc = "Optimizer class = RMS Prop", leave = False):
                self.forward_propagation(x)
                delta = self.back_propagation(y)
                for key in gradients.keys():
                    gradients[key] += delta[key]
                

                if (id+1) % self.batch_size == 0:
                    for key in lookahead.keys():
                        lookahead[key] = self.beta * _global[key] + self.learning_rate * gradients[key]
                    
                    for key in m_gradients.keys():
                        m_gradients[key] = self.beta1 * m_gradients[key] + (1 - self.beta1) * gradients[key]
                    
                    for key in v_gradients.keys():
                        v_gradients[key] = self.beta2 * v_gradients[key] + (1 - self.beta2) * (gradients[key]**2)
                    
                    for key in _parameters.keys():
                        m_gradients_hat[key] = m_gradients[key] / (1 - self.beta1 ** (epoch + 1))
                        v_gradients_hat[key] = v_gradients[key] / (1 - self.beta2 ** (epoch + 1))

                    
                    for key in _parameters.keys():
                        _parameters[key] = _parameters[key] - (self.learning_rate * m_gradients_hat[key]) / np.sqrt(v_gradients_hat + self.epsilon)
                    
                    for key in _global.keys():
                        _global[key] = lookahead[key]

                    
                    gradients = self.gradsInit()
            
        validation_yhat, validation_y, validation_acc, validation_loss = self.generateMetrics(self.x_val, self.y_val)
        print(f"Validation Accuracy: {validation_acc}", end = " ")
        print(f"Validation Loss: {validation_loss}", end = "\n")


        train_yhat, train_y, train_acc, train_loss = self.generateMetrics(self.x_val, self.y_val)
        print(f"Train Accuracy: {train_acc}", end = "")
        print(f"Train Loss: {train_loss}", end = "\n")

        test_yhat, test_y, test_acc, test_loss = self.generateMetrics(self.x_test, self.y_test)
        print(f"Test Accuracy: {test_acc}", end = " ")
        print(f"Test Loss: {test_loss}", end = "\n")

        return _parameters
    

    