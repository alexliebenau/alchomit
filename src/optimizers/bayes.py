from base import optimizer, ParallelEvaluator
from search_based import random_search, grid_search, staged_search
from evo import evolutionary
from gradient import bfgs
from ..models import gaussian_process, custom, get_x
import numpy as np
from scipy.stats import norm


class bayesian(optimizer):
    def __init__(self, function_approximator=None, acq_func: str = 'EI',
                 acq_optimizer: str | optimizer = None, early_stopping: bool = True, kappa: float = 1., *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function_approximator = function_approximator if function_approximator is not None else gaussian_process(use_gpu=True)
        # usually a gaussian process to approximate simulation model
        self.acq = acq_func
        self.early_stopping = early_stopping
        self.bounds = None  # will be set when run() is called
        self.optimizer_alg = 'grid_search' if acq_optimizer is None else acq_optimizer  # also set in run()
        self.minimizer = None  # will be set during run(

        # controls exploration-exploitation trade-off
        self.kappa = kappa

    def build_minimizer(self):

        obj_func = lambda x: self.surrogate(x)
        objective_model = custom(obj_func)

        if isinstance(self.optimizer_alg, str):
            sample_space_lenghts = [len(s) for s in self.framework.decision_variables.get_sample_space()]
            grid = grid_search(objective_model=get_x, sweep=True, framework=self.framework).generate_parameter_grid()
            max_iter = len(grid)

            def bit_lenght(n):
                bits = 0
                while n >> bits: bits += 1
                return bits

            match self.optimizer_alg:
                case 'bfgs':
                    opt = bfgs(
                        iterations=1000,
                        objective_model=objective_model,
                        use_constraints=True
                    )

                case 'grid_search':
                    opt = grid_search(objective_model=objective_model, sweep=True, framework=self.framework)

                case 'random_search':
                    iterations = max_iter // 10 + bit_lenght(max_iter // 10) + 2
                    opt = random_search(
                        iterations=iterations,
                        objective_model=objective_model,
                        framework=self.framework
                    )

                case 'staged':
                    grid_iterations = np.round([sample_space_lenghts[i] / 5 for i in range(len(sample_space_lenghts))])
                    random_iterations = max_iter // 5 + bit_lenght(max_iter // 5)
                    opt = staged_search(
                        grid_search_iterations=grid_iterations,
                        random_search_iterations=random_iterations,
                        framework=self.framework
                    )

                case 'evolutionary':
                    size_individual = np.sum([bit_lenght(l) for l in sample_space_lenghts])
                    multiplier = bit_lenght(size_individual)
                    opt = evolutionary(
                        iterations=3 * multiplier,
                        objective_model=objective_model,
                        size_population=25 * multiplier,
                        XOPB=0.9,
                        MUTPB=(0.5, 1),
                        framework=self.framework
                    )

                case _:
                    raise ValueError(f'Invalid option for acq fxn optimizer: {self.optimizer_alg}')

        elif isinstance(self.optimizer_alg, optimizer):
            opt = self.optimizer_alg
            opt.set_framework(self.framework)
            opt.parallel_evaluator = ParallelEvaluator(objective_model, opt.n_workers)
            opt.model = objective_model

        else:
            raise TypeError(f'Invalid type for optimizer specified: {type(self.optimizer_alg)}.\n'
                            f'Set either string or optimizer object.')


        if len(self.framework.constraints) > 0:
            opt.constrained = True

        def min_fxn():
            opt_x, _ = opt.minimize()
            return opt_x

        return min_fxn

    def surrogate(self, x: np.ndarray) -> float:
        # includes acquisition function!
        y_mean, y_var = self.function_approximator.predict_mean_and_variance(x)
        try:
            y_min = np.amin(self.database['y'][np.logical_not(np.isnan(self.database['y']))])
        except ValueError:
            y_min = 0
        diff = y_mean - y_min

        match self.acq:
            case "EI":  # Expected improvement
                if y_var <= 0.0:
                    return 0.0

                z = diff / y_var
                # use scipy.stats module to compute cdf and pdf
                ei = diff * norm.cdf(z) + self.kappa * y_var * norm.pdf(z)
                return -ei
            case "UCB":  # Upper confidence bound
                ucb = diff + self.kappa * 1.96 * np.sqrt(y_var)
                return ucb
            case "LCB":  # Lower confidence bound
                lcb = diff - self.kappa * 1.96 * np.sqrt(y_var)
                return lcb
            case _:
                raise RuntimeError("Unknown acquisition function specified: ", self.acq)

    def step(self, i):
        print(f'iteration {i} started')
        # update model with data from last generation
        if i > 0:
            if i > 1:
                x_in = self.database['x'][:i, :].reshape(i, self.len_x)
                y_in = self.database['y'][:i, :].reshape(i, self.len_y)
                self.function_approximator.update_model(x_in, y_in)
                # self.function_approximator.initialize((x_in, y_in)
            else:
                self.function_approximator.update_model(self.database['x'][i - 1].reshape(1, -1),
                                                        self.database['y'][i - 1].reshape(1, -1))
                # self.function_approximator.initialize(self.database['x'][i - 1].reshape(1, -1),
                #                                       self.database['y'][i - 1].reshape(1, -1))
        # print(self.function_approximator.model.model.)
        # minimize acquisition function
        from timeit import default_timer
        start = default_timer()
        opt_x = self.minimizer()
        print(f'optx {opt_x}     time taken for opt: {start-default_timer()}')
        # check if optimization stopped moving, if yes -> early stopping
        if self.early_stopping and i > 1 and np.array_equal(opt_x, self.database['x'][i - 1]):
            opt_y = None
        else:
            opt_y = self.model.predict(opt_x)
        return opt_x, opt_y

    def run(self) -> ([np.ndarray.dtype, np.ndarray.dtype], np.ndarray.dtype):
        # initialize bounds and random value generator
        # self.bounds = np.concatenate((self.lb, self.ub), axis=1)

        # set minimizer for acquisition function
        self.minimizer = self.build_minimizer()

        # initialize GPR if no data has been given yet
        if self.function_approximator.model.model is None:
            x_in, y_in = random_search(iterations=1,
                                       framework=self.framework,
                                       objective_model=self.model).run()
            print(x_in)
            self.function_approximator.initialize(x_in.reshape(1, -1), y_in.reshape(1, -1))

        # main loop
        for i in range(self.n_iter):
            x, y = self.step(i)
            print(self.len_x, x)
            self.database[i] = (x, y)
            # check if optimization stopped moving, if yes -> early stopping
            if self.early_stopping and i > 1 and np.array_equal(self.database['x'][i], self.database['x'][i - 1]):
                print("Early stopping of Bayesian Optimisation due to stagnation. Stopped at iteration ", i)
                self.database = np.delete(self.database, range(i, self.n_iter), axis=0)
                break

        index_y_min = np.argmin(self.database['y'])
        opt_x = self.database['x'][index_y_min]
        opt_y = self.database['y'][index_y_min]
        return opt_x, opt_y