from ..base import model
from ..preprocessing import scaler, pca
import numpy as np
import gpflow
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.cluster.vq import kmeans


class gaussian_process(model):
    def __init__(self, x_in: np.ndarray = None, y_in: np.ndarray = None, kernel=None, use_gpu: bool = False,
                 sparse: bool = False, reduce_dims: bool | float = False, scale: bool = False):

        if reduce_dims:
            scale = True

        self.scale = scale
        self.reduce_dims = reduce_dims
        self.x_scaler = None
        self.y_scaler = None
        self.pca = None

        if use_gpu:
            self.model = gpflow_gp(x_in, y_in, kernel, sparse)
        else:
            self.model = sklearn_gp(x_in, y_in, kernel)

        if x_in is not None and y_in is not None:
            self.initialize(x_in, y_in)

    def predict(self, x_in):
        # use model to make a prediction and return only mean value
        if self.scale:
            x_in = self.x_scaler.scale(x_in)

        if self.reduce_dims:
            x_in = self.pca.reduce_dimensions(x_in)

        res = self.model.predict(x_in.reshape(1, -1))
        if self.scale:
            res = self.y_scaler.scale_back(res)

        return res

    def predict_mean_and_variance(self, x_in):
        if self.scale:
            x_in = self.x_scaler.scale(x_in)

        if self.reduce_dims:
            x_in = self.pca.reduce_dimensions(x_in)

        mean, var = self.model.predict_mean_and_variance(x_in.reshape(1, -1))

        if self.scale:
            mean = self.y_scaler.scale_back(mean)
            var = mean * var

        return mean, var

    def update_model(self, new_x: np.ndarray, new_y: np.ndarray):
        self.model.update_model(new_x, new_y)

    def initialize(self, x_in, y_in, kernel=None, sparse=None):
        if self.scale:
            self.x_scaler = scaler(x_in)
            self.y_scaler = scaler(y_in)
            x_in = self.x_scaler.scale_data(x_in)
            y_in = self.y_scaler.scale_data(y_in)

        if self.reduce_dims:
            explained_variance_ratio = 0.95 if isinstance(self.reduce_dims, bool) else self.reduce_dims
            self.pca = pca(x_in, explained_variance_ratio)
            x_in = self.pca.reduce_data(x_in)
        self.model.initialize(x_in, y_in, kernel, sparse)


class sklearn_gp(model):
    def __init__(self, x_in: np.ndarray = None, y_in: np.ndarray = None, kernel=None):
        self.model = GaussianProcessRegressor(
            kernel=kernel if kernel is not None else Matern(),
            random_state=42
        )

        # create Gaussian Process Regression model if data is given
        if x_in is not None and y_in is not None:
            self.initialize(x_in, y_in)

    def predict(self, x_in):
        # use model to make a prediction and return only mean value
        return self.model.predict(x_in.reshape(1, -1))

    def predict_mean_and_variance(self, x_in):
        mean, std_dev = self.model.predict(x_in.reshape(1, -1), return_std=True)
        variance = std_dev ** 2
        return mean, variance

    def update_model(self, new_x: np.ndarray, new_y: np.ndarray):
        self.model = self.model.fit(new_x, new_y)

    def initialize(self, x_in, y_in, kernel=None, sparse=None):
        self.update_model(x_in, y_in)


class gpflow_gp(model):
    def __init__(self, x_in: np.ndarray = None, y_in: np.ndarray = None, kernel=None, sparse: bool = False):

        # if nothing specified, use Matern52 Kernel
        if kernel is None:
            kernel = gpflow.kernels.Matern52()

        self.kernel = kernel
        self.model = None
        self.sparse = sparse

        # create Gaussian Process Regression model if data is given
        if x_in is not None and y_in is not None:
            self.initialize(x_in, y_in)

    def predict(self, x_in):
        # use model to make a prediction and return only mean value
        y, _ = self.model.predict_y(x_in.reshape(1, -1))
        return y

    def predict_mean_and_variance(self, x_in):
        return self.model.predict_y(x_in.reshape(1, -1))

    def predict_f(self, x):
        return self.model.predict_f(np.array([x]))

    def update_model(self, new_x: np.ndarray, new_y: np.ndarray):
        if self.sparse:
            self.model = gpflow.models.SGPR(data=(new_x, new_y),
                                            kernel=self.kernel,
                                            inducing_variable=find_inducing_points(300, new_x))
        else:
            self.model = gpflow.models.GPR(data=(new_x, new_y),
                                           kernel=self.kernel)

    def initialize(self, x_in, y_in, kernel=None, sparse=None):

        self.update_model(x_in, y_in)
        # train model using optimizer
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.model.training_loss, self.model.trainable_variables)


def find_inducing_points(n_points, dataset):
    inducing_points, _ = kmeans(dataset, n_points)
    return inducing_points