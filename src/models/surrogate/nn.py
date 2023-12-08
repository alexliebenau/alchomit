from ..base import model
from ..preprocessing import scaler, pca
from generate_neural_network import neural_generator
import os


class neural_network(model):
    def __init__(self, X, Y,    # compulsory
                 model=None, epochs: int = None, batch_size: int = None,
                 k=5,  # for k-fold cross validation
                 max_layers=5,  # num hidden layers
                 max_probability_dropout=0.5,
                 activation_fxns=None,  # default: ['relu', 'tanh', 'sigmoid', 'elu']
                 max_units_per_layer=8,  # 2**n, e.g. 8 --> 256
                 max_l1=0.01, max_l2=0.01, max_momentum=0.9, max_learning_rate=0.9, max_epochs=300,
                 max_batch_size=7, # 2**n, e.g. 8 --> 256
                 reduce_dims: bool | float = False
                 ):

        self.scale_x = scaler(X)
        self.scale_y = scaler(Y)
        x_scaled = self.scale_x.scale_data(X)
        y_scaled = self.scale_y.scale_data(Y)

        if reduce_dims:
            self.reduce_dims = True
            explained_variance_ratio = 0.95 if isinstance(reduce_dims, bool) else reduce_dims
            self.pca = pca(x_scaled, explained_variance_ratio)
            x_scaled = self.pca.reduce_data(x_scaled)
        else:
            self.reduce_dims = False

        self.neuralnet = neural_generator(x_scaled, y_scaled, k, max_layers, max_probability_dropout,
                                          activation_fxns,
                                          max_units_per_layer, max_l1, max_l2, max_momentum, max_learning_rate,
                                          max_epochs,
                                          max_batch_size)

        if model is None:
            self.model = self.neuralnet.generate_model()
            self.model = self.neuralnet.train(self.model)
        else:
            self.model = model
            self.model = self.neuralnet.train(self.model, epochs=epochs, batch_size=batch_size)

    def plot_loss(self, history=None):
        self.neuralnet.plot_loss(history)

    def save(self, model=None, name=''):
        if model is None:
            model = self.model
        # save model to disk
        model_name = 'optimized_nn_' + name + str(hash(self.neuralnet))
        path = os.getcwd()
        model._name = model_name
        model.save(os.path.join(path, model_name))

    def predict(self, x):
        x_scaled = self.scale_x.scale(x)

        if self.reduce_dims:
            x_scaled = self.pca.reduce_dimensions(x_scaled)

        y_scaled = self.model.predict(x_scaled.reshape(1, -1), use_multiprocessing=True, workers=10)
        y = self.scale_y.scale_back(y_scaled)
        return y