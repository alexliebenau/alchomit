### ------------------------------------------------------------------ ###
#                          Superclass for all                            #
### ------------------------------------------------------------------ ###
from multiprocessing import Pool, Manager
from ..core import framework as fw
from ..models import simulation
from copy import deepcopy
import numpy as np

class ParallelEvaluator:
    def __init__(self, instance_model, num_workers=None):
        if num_workers == -1:
            self.num_workers = None
        else:
            self.num_workers = num_workers
        self.manager = Manager()
        self.model = instance_model
        self.model_instances = self.manager.dict()

    def evaluate(self, objective_function, x_values):
        with self.init_pool() as pool:
            results = pool.map(objective_function, x_values)
        return np.array(results)

    def init_pool(self):
        return Pool(self.num_workers, initializer=self.initialize_model, initargs=[self.model])

    @staticmethod
    def initialize_model(model_instance):
        if not isinstance(model_instance, simulation):
            global worker_model
            worker_model = deepcopy(model_instance)


class optimizer:
    def __init__(self, iterations: int, objective_model, framework: fw = None,
                 use_scipy: bool = False, parallel: bool | int = False, log: bool = False):
        self.n_iter = iterations
        # model needs to implement 'update_model' and 'predict_mean_and_var' methods!
        self.model = objective_model

        # constraint flag, set true when initializing
        self.constrained = False

        # set flags for using scipy or running in parallel
        self.scipy = use_scipy

        if use_scipy and log:
            log = False
            print(f'WARNING: Logging disabled when using SciPy optimizers. '
                  f'Set use_scipy=False in case logging is needed.')

        # set logging flag
        self.log = log
        self.parallel = parallel

        # get number of parallel workers
        if self.parallel:
            if isinstance(self.parallel, int):
                self.n_workers = self.parallel
            else:
                self.n_workers = -1
        else:
            self.n_workers = 1

        self.parallel_evaluator = ParallelEvaluator(num_workers=self.n_workers, instance_model=self.model)

        if framework is None:
            self.framework = None
            # create numpy arrays for x1, x2 and y
            self.len_y = None
            self.len_x = None
            if self.log:
                self.database = None

            # store lower and upper bounds and constraint checker:
            self.lb = None
            self.ub = None
            self.bounds = None
        else:
            self.set_framework(framework)

    def set_framework(self, framework):
        self.framework = framework
        # create numpy arrays for x1, x2 and y
        self.len_y = len(self.framework.objective_variables)
        self.len_x = len(self.framework.decision_variables)

        # build database
        if self.log:
            dt = np.dtype([('x', float, (self.len_x,)), ('y', float, (self.len_y,))])
            self.database = np.full(self.n_iter, np.nan, dtype=dt)

        # store lower and upper bounds and constraint checker:
        self.lb = self.framework.decision_variables.get_lower_bound()
        self.ub = self.framework.decision_variables.get_upper_bound()
        self.bounds = [(self.lb[i], self.ub[i]) for i in range(len(self.framework.decision_variables))]
        self.model.variable_array = self.framework.decision_variables

    def evaluate(self, objective_function, x_values):
        if self.parallel:
            result = self.parallel_evaluator.evaluate(objective_function, x_values)
        else:
            result = np.empty((len(x_values), self.len_y))
            for index, value in enumerate(x_values):
                result[index] = self.model.predict(value)

        if self.log:
            self.database['x'][:self.n_iter] = x_values
            self.database['y'][:self.n_iter] = result

            # Find empty rows and delete them
            nan_rows = np.isnan(self.database['x']).all(axis=1)
            self.database = self.database[~nan_rows]

        index_y_min = np.argmin(result)
        opt_x = x_values[index_y_min]
        opt_y = result[index_y_min]
        return opt_x, opt_y

    def run(self) -> ([np.ndarray.dtype, np.ndarray.dtype], np.ndarray.dtype):
        raise NotImplementedError(f'WARNING: Using  not implemented for this type of optimisation.')

    def run_scipy(self) -> ([np.ndarray.dtype, np.ndarray.dtype], np.ndarray.dtype):
        raise NotImplementedError(f'WARNING: Using SciPy not implemented for this type of optimisation.')

    def minimize(self):
        if self.framework is None:
            raise AttributeError('No framework found. Please specify framework using set_framework method!')
        if self.scipy:
            opt_x, opt_y = self.run_scipy()
        else:
            opt_x, opt_y = self.run()
        return opt_x, opt_y