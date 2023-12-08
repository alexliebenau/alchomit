from base import optimizer, ParallelEvaluator
from ..models import get_x
import numpy as np
from scipy.optimize import brute

### ------------------------------------------------------------------ ###
#                           Random Search                                #
### ------------------------------------------------------------------ ###

class random_search(optimizer):
    def __init__(self, mean: np.ndarray = None, variance: np.ndarray = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.scipy:
            print(f'WARNING: Using SciPy is not implemented for this kind of optimisation.')
            self.scipy = False

        if mean is not None:
            self.mean = self.framework.decision_variables.get_indices_from_values(mean)
            if variance is not None:
                self.variance = variance
            else:
                self.variance = np.ones(mean.shape)

    def get_random_values(self) -> np.ndarray.dtype:
        # check if has hung up
        generator_counter = 0

        while generator_counter < 1e+07:
            # generate random values and make sure they are within constraints
            if hasattr(self, 'mean'):
                found = False
                while not found:
                    x = np.random.normal(self.mean, np.sqrt(self.variance))
                    if (self.lb <= x).all() and (x <= self.ub).all():
                        found = True
                out = self.framework.round_indices(x)
            else:
                x = np.random.uniform(self.lb, self.ub)
                out = self.framework.round_indices(x)

            # check if is in bounds, if not generate new numbers
            if self.framework.check_constraints(out):
                return out
            generator_counter += 1

        raise RuntimeError('Aborting. The constraints are either set too tight or impossible to fulfill resulting '
                           'in a never ending loop')

    def run(self) -> ([np.ndarray.dtype, np.ndarray.dtype], np.ndarray.dtype):
        points = np.empty((self.n_iter, self.len_x))
        for index in range(self.n_iter):
            points[index] = self.get_random_values()
        worker_model = self.model
        x_opt, y_min = self.evaluate(worker_model.predict, points)
        return x_opt, y_min

    def generate_x(self):
        previous_model = self.model
        self.model = get_x()
        x_array = []
        for i in range(self.n_iter):
            x, _ = self.run()
            x_array.append(x)
        self.model = previous_model
        return np.array(x_array)


### ------------------------------------------------------------------ ###
#                             Grid Search                                #
### ------------------------------------------------------------------ ###

class grid_search(optimizer):
    def __init__(self, iterations_per_input: list | np.ndarray = None, sweep: bool = False, *args, **kwargs):

        # will be set when initializing framework
        self.x = None
        self.y = None
        self.x_iter = None
        self.iterations_per_input = iterations_per_input
        self.parameter_grid = None
        self.sweep = sweep

        # will be set when generating fxn for param grid is called
        self.parameter_grid = []
        self.x_iter = None

        # if iterations_per_input is None and 'iterations' not in kwargs:
        if 'iterations' not in kwargs:
            # Evenly distribute iterations among input variables
            super().__init__(iterations=1, *args, **kwargs)

        else:
            # iterations will be overridden when initializing framework
            super().__init__(*args, **kwargs)

    def set_iterations_per_input(self):
        if self.sweep:
            self.iterations_per_input = [len(var.sample_space) for var in self.framework.decision_variables]

        if self.iterations_per_input is None:
            # Evenly distribute iterations among input variables
            dim_sample_space = np.zeros(self.len_x)
            for i, var in enumerate(self.framework.decision_variables):
                # if var.val_type == 'real':
                #     dim_sample_space[i] = 100
                # else:
                dim_sample_space[i] = len(var.sample_space)
            norm_iterations = dim_sample_space / np.sum(dim_sample_space) * self.n_iter
            factor = (self.n_iter / np.prod(norm_iterations)) ** (1 / self.len_x)
            self.iterations_per_input = np.round(norm_iterations * factor).astype(int)
        else:
            self.n_iter = np.prod(self.iterations_per_input)

        if self.log:
            dt = np.dtype([('x', float, (self.len_x,)), ('y', float, (self.len_y,))])
            self.database = np.full(self.n_iter, np.nan, dtype=dt)

    def set_framework(self, framework):
        super().set_framework(framework)
        self.parameter_grid = self.generate_parameter_grid()

    def objective_function(self, x):
        if self.framework.check_constraints(x):
            return self.model.predict(x)
        return np.inf

    def run_scipy(self):
        ranges = []
        for i in range(len(self.framework.decision_variables)):
            ranges.append(slice(self.lb[i], self.ub[i], self.iterations_per_input[i]))
        ranges = tuple(ranges)

        opt = brute(
            func=self.objective_function,
            ranges=ranges,
            finish=None,
            workers=self.n_workers,
            full_output=True
        )

        opt_x, opt_y = opt[0], opt[1]
        print(self.iterations_per_input)
        return opt_x, opt_y

    def run(self):
        self.n_iter = len(self.parameter_grid)
        worker_model = self.model
        x_opt, y_min = self.evaluate(worker_model.predict, self.parameter_grid)
        return x_opt, y_min

    def generate_parameter_grid(self):
        self.x_iter = np.array([self.lb[i] for i in range(self.len_x)]).astype(float)
        self.set_iterations_per_input()

        ### build grid array and corresponding zero-array for y
        # Initialize the parameter and search range
        search_range = []

        # Build search range
        for index in range(self.len_x):
            search_step = ((self.ub[index] - self.lb[index]) / (self.iterations_per_input[index] - 1))
            search_step = self.framework.decision_variables[index].transform(search_step)
            search_range.append(np.arange(start=self.lb[index],
                                          stop=self.ub[index],
                                          step=search_step))

        # Start the grid search
        parameter_grid = []

        def grid_search_recursive(index, search_range):
            if index >= self.len_x:
                # Base case: All dimensions have been iterated
                if self.framework.check_constraints(self.x_iter):
                    parameter_grid.append(self.x_iter.copy())
            else:
                # Recursive case: Iterate over the search range for the current dimension
                for x in search_range[index]:
                    self.x_iter[index] = x
                    grid_search_recursive(index + 1, search_range)

        grid_search_recursive(0, search_range)
        return np.array(parameter_grid)

### ------------------------------------------------------------------ ###
#                            Staged Search                               #
### ------------------------------------------------------------------ ###

class staged_search(optimizer):
    def __init__(self, grid_search_iterations: int | list | np.ndarray, random_search_iterations: int, *args, **kwargs):
        super().__init__(iterations=1, *args, **kwargs)

        found = False
        if 'log' in kwargs:
            found = True
            log = kwargs['log']
            kwargs.pop('log')

        if isinstance(grid_search_iterations, int) | isinstance(grid_search_iterations, np.number):
            self.gs = grid_search(iterations=grid_search_iterations, log=True, *args, **kwargs)
        elif isinstance(grid_search_iterations, list) | isinstance(grid_search_iterations, np.ndarray):
            self.gs = grid_search(iterations_per_input=grid_search_iterations, log=True, *args, **kwargs)

        # disable parallel execution in random search
        if 'parallel' in kwargs:
            kwargs.update({'parallel': False})

        # disable scipy execution in random search
        if 'use_scipy' in kwargs:
            print(f'Notice: Random search has no SciPy implementation.')
            kwargs.update({'use_scipy': False})

        if found:
            kwargs.update({'log': log})

        self.rs = random_search(iterations=random_search_iterations, *args, **kwargs)

    def run(self):
        self.gs.set_framework(self.framework)
        self.gs.model = self.model
        self.parallel_evaluator = ParallelEvaluator(self.model, 1)
        self.rs.set_framework(self.framework)
        self.rs.model = self.model
        self.parallel_evaluator = ParallelEvaluator(self.model, 1)

        gs_opt_x, _ = self.gs.minimize()
        num_avg_values = 5 if len(self.gs.database['y']) >= 6 else len(self.gs.database['y']) - 1
        smallest_indices = np.argpartition(self.gs.database['y'].reshape(1, -1), num_avg_values)[0, :num_avg_values]
        smallest_x_values = np.empty((num_avg_values, self.len_x))
        for index, value in enumerate(smallest_indices):
            smallest_x_values[index] = self.gs.database['x'][value]

        if self.log:
            self.database = self.gs.database

        variance = np.var(smallest_x_values, axis=0)
        self.rs.mean = gs_opt_x
        self.rs.variance = variance

        opt_x, opt_y = self.rs.minimize()
        if self.log:
            self.database = np.concatenate((self.database, self.rs.database))
        return opt_x, opt_y