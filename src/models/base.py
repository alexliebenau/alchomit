import numpy as np

# define interface for subclasses
class model:
    def predict(self, x) -> np.ndarray:
        raise NotImplementedError('Method "predict" not implemented for class {}'.format(type(self)))

    def predict_mean_and_variance(self, x) -> (np.ndarray, np.ndarray):
        raise NotImplementedError('Method "predict_mean_and_variance" not implemented for class {}'.format(type(self)))

    def step(self, x_in, i) -> (np.ndarray, np.ndarray):
        return self.predict(x_in)

    def update_model(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError('Method "update_model" not implemented for class {}'.format(type(self)))

    def initialize(self, x_in, y_in, framework):
        raise NotImplementedError('Method "initialize" not implemented for class {}'.format(type(self)))


### ------------------------------------------------------------------ ###
#                         Generate input data (x)                        #
### ------------------------------------------------------------------ ###
class get_x(model):
    def predict(self, x) -> np.ndarray:
        return np.empty(1)

    def predict_mean_and_variance(self, x) -> (np.ndarray, np.ndarray):
        return np.empty(1), np.empty(1)


### ------------------------------------------------------------------ ###
#                         Generate input data (x)                        #
### ------------------------------------------------------------------ ###
class custom(model):
    def __init__(self, obj_fxn):
        self.obj_fxn = obj_fxn

    def predict(self, x) -> np.ndarray:
        return self.obj_fxn(x)

    def predict_mean_and_variance(self, x) -> (np.ndarray, np.ndarray):
        return self.obj_fxn(x)