import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.stats import norm
import gpflow
import random as rd


class neural_generator:
    def __init__(self, X, Y, k,
                max_layers, max_probability_dropout, activations,
                max_units_per_layer, max_l1, max_l2, max_momentum,
                max_learning_rate, max_epochs, max_batch_size):

        # store data set
        self.X = X
        self.Y = Y

        # store hyperparameter variables
        self.max_layers = max_layers

        if activations is None:
            self.activations = ['relu', 'tanh', 'sigmoid', 'elu']
        else:
            self.activations = activations

        self.arch_fw = lambda i: _architecture_framework(
            folds=KFold(n_splits=k, shuffle=True, random_state=69),
            X=X, Y=Y,
            n_layers=i,
            max_probability_dropout=max_probability_dropout,
            activations=self.activations,
            max_units_per_layer=max_units_per_layer,
            max_l1=max_l1,
            max_l2=max_l2,
            max_momentum=max_momentum,
            max_learning_rate=max_learning_rate,
            max_epochs=max_epochs,
            max_batch_size=max_batch_size
        )

        self.epochs = 0
        self.batch_size = 0
        self.history = None

    def generate_model(self):
        # create a new neural network with optimized structure

        best_candidate = ([], float('inf'))

        for i in range(1, self.max_layers + 1):
            afw = self.arch_fw(i)

            # create initial array for GPR in Bayesian
            x_init, y_init = random_search(framework=afw,
                                           iterations=i,
                                           opt='Custom',
                                           custom_func=afw.predict
                                           ).run()

            # hit it babyy
            x, y = bayesian(framework=afw,
                            iterations=20 + 2 * len(afw.objective_vars),
                            opt='GPR',
                            acq_func='EI',
                            init_array=(x_init, y_init),
                            constraints=None,
                            optimizer_alg='L-BFGS-B',
                            custom_sim=afw,
                            early_stopping=True
                            ).run()

            # get best y
            y_min = min(y)
            if y_min < best_candidate[1]:
                index_y_min = np.where(y == y_min)[0]
                x_opt = x[index_y_min]
                best_candidate = (x_opt, y_min)

        # build best performing model
        x_opt = best_candidate[0][0]
        learning_rate, \
        momentum, \
        epochs, \
        batch_size, \
        layer_units, \
        activation_index, \
        dropout_probability, \
        l1, \
        l2 = afw.get_model_parameters(x_opt)

        # define optimizer
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum
        )

        # create model
        model = afw.build_model(n_dense=(len(x_opt) - 4) % 5,
                                layer_units=layer_units,
                                dropout_prob=dropout_probability,
                                l1_penalties=l1,
                                l2_penalties=l2,
                                act_fxn_indexes=activation_index)

        # compile model
        model.compile(
            loss='mean_absolute_error',
            optimizer=optimizer,
            metrics=[tf.keras.metrics.mean_absolute_error])

        afw.show_model(model, epochs, batch_size)
        self.epochs = epochs
        self.batch_size = batch_size
        return model

    def train(self, model, X=None, Y=None, epochs=None, batch_size=None):
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size

        # split into test and validation data
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.25, random_state=69)

        self.history = model.fit(x_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(x_val, y_val))
        return model

    def plot_loss(self, history=None):
        if history is None:
            history = self.history

        # plot learning curve
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.plot(loss, 'b')
        plt.plot(val_loss, 'g')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.show()

    @staticmethod
    def test(model, x_test, y_test):
        scale_x_test, scale_y_test = MinMaxScaler(), MinMaxScaler()
        x_test, _ = scale_x_test.fit_transform(x_test), scale_y_test.fit_transform(y_test)

        # get prediction of test data
        y_nn = scale_y_test.inverse_transform(
            model.predict(x_test, use_multiprocessing=True, workers=10)
        )

        # plot results
        x = range(x_test.shape[0])

        fig, ax = plt.subplots()
        ax.plot(x, y_test, 'b--', label='Test data from GPR')
        ax.plot(x, y_nn, 'g--', label='Predicted output of NN')
        ax.plot(x, np.absolute(np.subtract(y_test, y_nn)), 'r:', label='Absolute error')
        ax.set_xlabel('Input index')
        ax.set_ylabel('Output')
        ax.set_title('Prediction accuracy on set of random datapoints')
        ax.legend(loc='upper right')
        plt.show()


class _architecture_framework():
    def __init__(self, folds, X, Y,
                 n_layers, max_probability_dropout, activations,
                 max_units_per_layer, max_l1, max_l2, max_momentum,
                 max_learning_rate, max_epochs, max_batch_size):

        self.X = X
        self.Y = Y
        self.folds = folds
        self.n_layers = n_layers
        min_probability_dropout = 0  # multiplied by 0.1
        self.activations = activations
        max_index_activations = len(activations) - 1
        min_index_activations = 0
        min_l1 = 0.
        min_l2 = 0.
        min_units_per_layer = 4  # 2**n --> e.g. 8 --> 256
        min_momentum = 0.
        min_learning_rate = 0.0001
        min_epochs = 0.1  # multiplied by 100
        min_batch_size = 3  # 2**n --> e.g. 8 --> 256

        self.upper_bounds = [max_learning_rate, max_momentum, max_epochs, max_batch_size]
        self.lower_bounds = [min_learning_rate, min_momentum, min_epochs, min_batch_size]

        # define ins and outs
        self.objective_vars = ["learning_rate",
                               "momentum",
                               "n_epochs",
                               "batch_size"]
        for i in range(1, n_layers + 1):
            self.upper_bounds.append(max_units_per_layer)
            self.upper_bounds.append(max_index_activations)
            self.upper_bounds.append(max_probability_dropout)
            self.upper_bounds.append(max_l1)
            self.upper_bounds.append(max_l2)
            self.lower_bounds.append(min_units_per_layer)
            self.lower_bounds.append(min_index_activations)
            self.lower_bounds.append(min_probability_dropout)
            self.lower_bounds.append(min_l1)
            self.lower_bounds.append(min_l2)
            self.objective_vars.append("units_per_layer")
            self.objective_vars.append("index_activation_fxn")
            self.objective_vars.append("probability_dropout")
            self.objective_vars.append("l1_penalty")
            self.objective_vars.append("l2_penalty")

        self.out_vars = ["output"]

    def constraint_ok(self, fuckoff):
        return True

    def predict(self, x):

        # Create pandas DataFrame to show values
        df = pd.DataFrame({
            'Objective Variable': self.objective_vars,
            'Lower Bound': self.lower_bounds,
            'Actual Value': x,
            'Upper Bound': self.upper_bounds
        })
        # df.index = None
        print(df)

        # get hyperparameters from x
        learning_rate, \
        momentum, \
        epochs, \
        batch_size, \
        layer_units, \
        activation_index, \
        dropout_probability, \
        l1, \
        l2 = self.get_model_parameters(x)

        # define optimizer
        optimizer = tf.keras.optimizers.legacy.SGD(
            learning_rate=learning_rate,
            momentum=momentum
        )

        # set callback to stop training in case no further improvement is achieved
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss",
                                                          min_delta=1e+08,
                                                          patience=25,
                                                          verbose=1,
                                                          start_from_epoch=150)

        scores = []
        # train model using k-fold cross validation

        for index_train, index_val in self.folds.split(self.X, self.Y):
            scale_x, scale_y = MinMaxScaler(), MinMaxScaler()
            x_train, x_val = scale_x.fit_transform(self.X[index_train]), scale_x.fit_transform(self.X[index_val])
            y_train, y_val = scale_y.fit_transform(self.Y[index_train]), scale_y.fit_transform(self.Y[index_val])

            # create model
            model = self.build_model(n_dense=self.n_layers,
                                     layer_units=layer_units,
                                     dropout_prob=dropout_probability,
                                     l1_penalties=l1,
                                     l2_penalties=l2,
                                     act_fxn_indexes=activation_index)

            # compile model
            model.compile(
                loss='mean_absolute_error',
                optimizer=optimizer,
                metrics=[tf.keras.metrics.mean_absolute_error])

            # train with current fold
            history = model.fit(x_train, y_train,
                                verbose=0,
                                epochs=epochs,
                                batch_size=batch_size,
                                # callbacks=[early_stopping],
                                validation_data=(x_val, y_val))

            # get validation loss of last epoch and append to score
            val_loss = history.history['val_loss'][-1]
            if np.isnan(val_loss):
                val_loss = np.ones(1)
            scores.append(val_loss)

        self.show_model(model=model, epochs=epochs, batch_size=batch_size)

        mean_val_loss = np.mean(scores)
        # return the mean val loss of each fold
        print('Mean validation loss of all folds: ', mean_val_loss)
        print('\n ---------------------------------------------------------- \n')
        return mean_val_loss

    def build_model(self, n_dense: float | int,
                    layer_units: list,
                    dropout_prob: list,
                    l1_penalties: list,
                    l2_penalties: list,
                    act_fxn_indexes: list):

        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(len(self.X[0]),)))

        for i in range(n_dense):
            model.add(tf.keras.layers.Dense(
                layer_units[i],
                activation=self.activations[int(act_fxn_indexes[i])],
                kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_penalties[i],
                                                              l2=l2_penalties[i])
            )
            )
            if dropout_prob[i] > 0:
                model.add(tf.keras.layers.Dropout(dropout_prob[i]))

        model.add(tf.keras.layers.Dense(len(self.Y[0])))
        model.build()
        return model

    def get_model_parameters(self, x: list | np.ndarray):
        learning_rate = x[0]
        momentum = x[1]
        epochs = int(x[2] * 100)
        batch_size = int(2 ** round(x[3]))
        layer_units = (2 ** np.round(x[4::5])).astype(int)
        activation_index = np.round(x[5::5]).astype(int)
        dropout_probability = np.round(x[6::5], decimals=2) * 0.1
        l1_penalty = x[7::5]
        l2_penalty = x[8::5]

        print('\n            # Layers : ', self.n_layers, '\n',
              '       Learning Rate: ', learning_rate, '\n',
              '            Momentum: ', momentum, '\n',
              '              Epochs: ', epochs, '\n',
              '          Batch Size: ', batch_size, '\n',
              '     Units in layers: ', layer_units, '\n',
              ' Dropout Probability: ', dropout_probability, '\n',
              'Activation functions: ', [self.activations[i] for i in activation_index], '\n',
              '        L1 penalties: ', l1_penalty, '\n',
              '        L2 penalties: ', l2_penalty, '\n'
              )

        return learning_rate, momentum, epochs, batch_size, \
               layer_units, activation_index, dropout_probability, l1_penalty, l2_penalty

    @staticmethod
    def show_model(model, epochs, batch_size):
        model.summary()

        print('------------------------------------------\n',
              'Epochs:     ', epochs, '\n',
              'Batch Size: ', batch_size, '\n',
              '------------------------------------------\n')


class gaussian_process:
    def __init__(self, framework, init_array):
        self.fw = framework

        # create Gaussian Process Regression model
        self.model = gpflow.models.GPR(
            init_array,
            kernel=gpflow.kernels.SquaredExponential(),
        )

        # train model using optimizer
        self.opt = gpflow.optimizers.Scipy()
        self.opt.minimize(self.model.training_loss, self.model.trainable_variables)

    def predict(self, x_in):
        # use model to make a prediction and return mean and variance
        return self.model.predict_y(np.array([x_in]))

    def predict_range(self, range_min, range_max, n_samples=100):
        # if not specified otherwise, sample 100 points from GP model
        # this returns upper and lower confidence bound from GP model
        out_mean = np.zeros(n_samples, dtype=np.float32)
        out_var = np.zeros(n_samples, dtype=np.float32)

        sample_space = np.linspace(range_min, range_max, n_samples)
        for i in range(n_samples):
            out_mean[i], out_var[i] = self.predict(sample_space[i])

        return out_mean, out_var


class optimizer:
    def __init__(self, iterations: int, framework, opt: str,
                 acq_func: str, init_array: np.ndarray, kappa: float, minim: bool,
                 custom_func=None):
        self.fw = framework
        self.n_iter = iterations
        self.opt = opt

        # store custom surrogate function if given
        if custom_func is not None:
            self.custom_func = custom_func

        # By default, minimize unless set to false --> then maximize
        self.min = minim

        # create numpy arrays for x1, x2 and y
        self.len_y = len(self.fw.out_vars)
        self.y = np.zeros((iterations, self.len_y))
        self.len_x = len(self.fw.objective_vars)
        self.x = np.zeros((iterations, self.len_x))

        # if GPR is used, create model
        if init_array is None:
            self.init_index = 0
        else:
            self.x = np.append(init_array[0], self.x, axis=0)
            self.y = np.append(init_array[1], self.y, axis=0)
            self.init_index = init_array[0].shape[0]

        self.acq = acq_func
        self.curr_iter = None

        # controls exploration-exploitation trade-off
        self.kappa = kappa

        # define surrogate model
        match opt:
            # case "Simulation":
            #     self.model = simulation(self.fw)
            case "GPR":
                self.model = gaussian_process(self.fw, init_array)
            case "NN":
                raise RuntimeError("Using Neural networks is not implemented yet")
            case "GetX" | "Custom":
                # create test dataset
                self.model = None
            case _:
                raise ValueError("Unknown function approximator specified. Check call of optimizer subclass")

        # get min/max from y
        self.y_max = lambda i: self.y[np.argmax(self.y[i])]
        self.y_min = lambda i: self.y[np.argmin(self.y[i])]

    def surrogate(self, x: np.ndarray) -> float:
        match self.opt:
            case "Simulation":
                self.model.set_parameters(x)
                res = self.model.execute_model(step_size=self.fw.step_size,
                                               stop_time=self.fw.stop_time,
                                               )[-1]

                return (res + self.fw.offset) * self.fw.scale
            case "GPR":
                # includes acqusition function!
                y_mean, y_var = self.model.predict(x)
                # some kind of Tensor conversion missing here --> add to func_approx.py

                if self.min:
                    diff = y_mean - self.y_min(self.curr_iter)
                else:
                    diff = y_mean - self.y_max(self.curr_iter)

                match self.acq:
                    case "EI":  # Expected improvement
                        if y_var <= 0.0:
                            return 0.0

                        z = diff / y_var
                        # use scipy.stats module to compute cdf and pdf
                        ei = diff * norm.cdf(z) + self.kappa * y_var * norm.pdf(z)

                        # if minimizing, return. If maximizing, return negative (--> minimizing a negative number = maximizing)
                        if self.min:
                            return ei
                        return -ei

                    case "UCB":  # Upper confidence bound
                        return diff + self.kappa * 1.96 * np.sqrt(y_var)
                    case "LCB":  # Lower confidence bound
                        return diff - self.kappa * 1.96 * np.sqrt(y_var)
                    case _:
                        raise RuntimeError("Unknown acquisition function specified: ", self.opt)

            case "NN":
                raise RuntimeError("Using Neural networks is not implemented yet")

            case "GetX":
                # returns 0 so a test dataset for x's can be created
                return 0

            case "Custom":
                return self.custom_func(x)

            case _:
                raise ValueError("Unknown function approximator specified. Check call of optimizer subclass")

    def get_y(self, index):
        # dummy function, will be overridden
        pass

    def run(self) -> ([np.ndarray.dtype, np.ndarray.dtype], np.ndarray.dtype):
        # Use surrogate function to find next data point using acquisition function
        for i in range(self.n_iter):
            self.get_y(i)

        return self.x[self.init_index:], self.y[self.init_index:]


class random_search(optimizer):
    def __init__(self, iterations: int, framework, opt: str,
                 acq_func=None, init_array=None, kappa=float(1), minim=True, custom_func=None):
        super().__init__(iterations, framework, opt, acq_func, init_array, kappa, minim, custom_func)

        # set correct return value for surrogate
        if minim:
            self.acq = "LCB"
        else:
            self.acq = "UCB"

    def get_random_values(self) -> np.ndarray.dtype:
        # generate random values and make sure they are within size constraint
        out = np.zeros(self.len_x)
        for i in range(self.len_x):
            # get random number between 0.0 and 1.0 and multiply with max size
            out[i] = rd.uniform(self.fw.lower_bounds[i], self.fw.upper_bounds[i])

        if self.fw.constraint_ok(out.sum()):
            return out
        return self.get_random_values()

    def get_y(self, i):
        x = self.get_random_values()
        self.x[i + self.init_index] = x
        self.y[i + self.init_index] = np.array([self.surrogate(x)])


class bayesian(optimizer):
    def __init__(self, iterations: int, framework, opt: str, constraints: LinearConstraint | NonlinearConstraint,
                 optimizer_alg='SLSQP', acq_func=None, init_array=None, kappa=float(1), minim=True, custom_func=None,
                 custom_sim=None, early_stopping=True):
        super().__init__(iterations, framework, opt, acq_func, init_array, kappa, minim, custom_func)

        # if custom_sim is None:
        #     self.sim = simulation(self.fw)
        # else:
        self.sim = custom_sim

        self.early_stopping = early_stopping

        self.constraints = constraints
        self.optimizer_alg = optimizer_alg

        upper = np.array(self.fw.upper_bounds).reshape(len(self.fw.upper_bounds), 1)
        lower = np.array(self.fw.lower_bounds).reshape(len(self.fw.lower_bounds), 1)
        self.bounds = np.concatenate((lower, upper), axis=1)

    def get_y(self, iteration):
        print('Iteration ', iteration, ' started')
        # If pre-trained data was specified as init_array, include it when building GP
        index = iteration + self.init_index

        # initialize GP
        self.model = gaussian_process(framework=self.fw,
                                      init_array=(self.x[:index], self.y[:index])
                                      )
        self.curr_iter = iteration

        # minimize acquisition function
        sample = minimize(fun=self.surrogate,
                          method=self.optimizer_alg,
                          x0=np.zeros(len(self.fw.objective_vars)),
                          bounds=self.bounds,
                          constraints=self.constraints
                          )
        opt_x = sample.x

        # store optimal x and y values of iteration
        self.x[index] = opt_x
        self.y[index] = self.sim.predict(opt_x)

    def run(self) -> ([np.ndarray.dtype, np.ndarray.dtype], np.ndarray.dtype):
        # Use surrogate function to find next data point using acquisition function
        for i in range(self.n_iter):
            self.get_y(i)
            # check if optimization stopped moving, if yes -> early stopping
            if i > 4:
                stop = True
                for j in range(1, 6):
                    if stop:
                        if not np.array_equal(self.x[i + self.init_index], self.x[i + self.init_index - j]):
                            stop = False
                if stop:
                    print("Early stopping of Bayesian Optimisation due to stagnation. Stopped at iteration ", i)
                    return self.x[self.init_index:i + self.init_index], self.y[self.init_index:i + self.init_index]
        return self.x[self.init_index:], self.y[self.init_index:]
