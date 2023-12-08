from base import controller
from ..models import model, custom, get_x
from ..optimizers import *
from ..optimizers import base as optbase
import numpy as np


class optimal(controller):
    def __init__(self, optimizer_alg: str | model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer_alg = optimizer_alg
        self.y = np.empty(self.len_y)
        self.s = np.empty(self.len_s)

    def predict(self, params=None):
        # parameter_grid = grid_search(
        #         sweep=True,
        #         objective_model=get_x,
        #         framework=self.framework).generate_parameter_grid()
        #
        # if self.search_algorithm == 'random':
        #     iters = len(parameter_grid) // 10
        #     search_points = random_search(
        #         iterations=iters if iters > 0 else 1,
        #         objective_model=get_x,
        #         framework=self.framework).generate_x()
        # elif self.search_algorithm == 'grid':
        #     search_points = parameter_grid
        #
        #
        # else:
        #     raise ValueError('Unknown search algorithm specified: ', self.search_algorithm)

        self.y, self.s = self.get_initial_state(params)
        index = 1

        def objfxn(x):
            return self.obj_func(x, index)

        objective_model = custom(objfxn)

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
                        objective_model=objective_model
                    )

                case 'staged':
                    grid_iterations = np.round([sample_space_lenghts[i] / 5 for i in range(len(sample_space_lenghts))])
                    random_iterations = max_iter // 5 + bit_lenght(max_iter // 5)
                    opt = staged_search(
                        grid_search_iterations=grid_iterations,
                        random_search_iterations=random_iterations
                    )

                case 'evo':
                    size_individual = np.sum([bit_lenght(l) for l in sample_space_lenghts])
                    multiplier = bit_lenght(size_individual)
                    opt = evolutionary(
                        iterations=3 * multiplier,
                        objective_model=objective_model,
                        size_population=25 * multiplier,
                        XOPB=0.9,
                        MUTPB=(0.5, 1)
                    )

                case _:
                    raise ValueError(f'Invalid option for acq fxn optimizer: {self.optimizer_alg}')

        elif isinstance(self.optimizer_alg, optbase.optimizer):
            opt = self.optimizer_alg
            opt.model = objective_model
            opt.parallel_evaluator = optbase.ParallelEvaluator(objective_model, opt.n_workers)
            print(f'been here')
            opt.set_framework(self.framework)
            print(opt.model)

        else:
            raise TypeError(f'Invalid type for optimizer specified: {type(self.optimizer_alg)}.\n'
                            f'Set either string or optimizer object.')

        opt.set_framework(self.framework)
        if len(self.framework.constraints) > 0:
            opt.constrained = True


        while index < self.n_iter:
            print(f'iteration: {index}')
            opt_x, delta_y = opt.minimize()
            self.y += delta_y
            self.stored_x[index] = opt_x

            if self.log:
                self.database['x'][index] = opt_x
                self.database['s'][index] = self.s
                self.database['y'][index] = self.y

            index += 1
        #
        # if isinstance(self.function_approximator, simulation):
        #     # get actual values
        #     print(f'   ...getting actual values')
        #     self.sim.write_inputs(x, self.framework.decision_variables)
        #     actual_y = self.sim.predict(params)
        # else:
        #     actual_y = y

        return self.y

    def obj_func(self, x_in, index):
        x_in = np.concatenate((x_in, self.s))
        delta_y, delta_s = self.function_approximator.step(x_in, index)
        s = self.s + delta_s
        y = delta_y
        return delta_y
