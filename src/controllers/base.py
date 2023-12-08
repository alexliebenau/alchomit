from ..models import model, simulation, gaussian_process
from ..core import framework as fw
from ..optimizers.base import ParallelEvaluator
from ..optimizers.search_based import random_search
import numpy as np


class controller(model):
    def __init__(self, sim: simulation, framework: fw = None, function_approximator: str = None,
                 reduce_dimensionality: bool = False, scale: bool = False, parallel: bool = False,
                 use_scipy: bool = False, log: bool = False):
        super().__init__()
        self.sim = sim
        self.n_iter = self.sim.iterations
        self.function_approximator = function_approximator
        # constraint flag, set true when initializing
        self.constrained = False
        self.scale = scale
        self.reduce_dims = reduce_dimensionality
        self.scipy = use_scipy
        if use_scipy and log:
            log = False
            print(f'WARNING: Logging disabled when using SciPy optimizers. '
                  f'Set use_scipy=False in case logging is needed.')
        self.log = log
        self.parallel = parallel

        # get number of parallel workers
        if self.parallel:
            if isinstance(self.parallel, int):
                self.n_workers = self.parallel
            else:
                self.n_workers = -1
        else:
            self.n_workers = None

        self.parallel_evaluator = ParallelEvaluator(num_workers=self.n_workers,
                                                    instance_model=self.function_approximator)
        self.stored_x = None
        if framework is None:
            self.framework = None
            # create numpy arrays for x1, x2 and y
            self.len_y = None
            self.len_x = None
            self.len_s = None
            self.len_u = None
            self.inputs = None
            self.csv_inputs = None
            if self.log:
                self.database = None
            self.training_set = None
            self.pca = None
            self.x_scaler = None
            self.y_scaler = None

            # store lower and upper bounds and constraint checker:
            self.lb = None
            self.ub = None
        else:
            self.set_framework(framework)

    def set_framework(self, framework):
        self.framework = framework
        # create numpy arrays for x1, x2 and y
        self.len_y = len(self.framework.objective_variables)
        self.len_x = len(self.framework.decision_variables)
        self.len_s = len(self.framework.state_variables)
        self.stored_x = np.empty((self.n_iter, self.len_x))

        # get input variable names
        self.inputs = self.sim.get_sim_variables()['input_vars']
        for objvar in self.framework.decision_variables.get_name():
            if objvar in self.inputs:
                self.inputs.remove(objvar)
        self.len_u = len(self.inputs)
        self.csv_inputs = self.sim.csv_input_file[self.inputs].fillna(0).to_numpy()[1:]

        # # set variable filter on simulation to ouput state vars as well
        # self.sim.set_variable_filter([self.framework.state_variables, self.framework.objective_variables])

        # build database
        if self.log:
            dt = np.dtype([('x', float, (self.len_x,)), ('s', float, (self.len_s,)), ('y', float, (self.len_y,))])
            self.database = np.empty(self.n_iter, dtype=dt)

        # store lower and upper bounds and constraint checker:
        self.lb = self.framework.decision_variables.get_lower_bound()
        self.ub = self.framework.decision_variables.get_upper_bound()
        self.sim.variable_array = self.framework.decision_variables

        if not isinstance(self.function_approximator, model):
            if self.function_approximator is None:
                self.function_approximator = self.sim
            else:
                data_x, data_y = self.generate_training_dataset(5)
                if self.function_approximator == 'GPR':
                    self.function_approximator = gaussian_process(data_x, data_y,
                                                                  reduce_dims=self.reduce_dims, scale=self.scale)
                elif self.function_approximator == 'sparse_GPR':
                    self.function_approximator = gaussian_process(data_x, data_y, sparse=True,
                                                                  reduce_dims=self.reduce_dims, scale=self.scale)
                elif self.function_approximator == 'NN':
                    raise NotImplementedError('NN as fxn approximators coming soon')
                else:
                    raise ValueError('Unknown function approximater specified: ', self.function_approximator)

    # def evaluate(self, objective_function, x_values, s_values):
    #     if self.parallel:
    #         result = self.parallel_evaluator.evaluate(objective_function, x_values)
    #     else:
    #         result = np.empty((len(x_values), self.len_y))
    #         for index, value in enumerate(x_values):
    #             result[index] = self.function_approximator.predict(value)
    #
    #     if self.log:
    #         self.database['x'] = x_values
    #         self.database['y'] = result
    #
    #     index_y_min = np.argmin(result)
    #     opt_x = x_values[index_y_min]
    #     opt_y = result[index_y_min]
    #     return opt_x, opt_y

    def generate_training_dataset(self, iterations: int):
        print('generate dataset')
        # generate dataset with each time step from simulation using random values for ctrl vars
        dt = np.dtype([('x', float, (self.len_x,)), ('s', float, (self.len_s,)), ('y', float, (self.len_y,))])
        iteration_results = np.empty(self.n_iter, dtype=dt)
        tmp_arr = []
        for i in range(iterations):
            tmp_arr.append(iteration_results)
        dataset = np.array(tmp_arr)
        for index in range(iterations):
            for j in range(self.n_iter):
                iteration_results['x'][j] = random_search(iterations=1,
                                                          objective_model=self.sim,
                                                          framework=self.framework).get_random_values()
            self.sim.write_inputs(iteration_results['x'])
            res = self.sim.execute_model(variables='')
            iteration_results['s'] = res[1:self.n_iter + 1, 1:self.len_s + 1]
            iteration_results['y'] = res[1:self.n_iter + 1, -self.len_y:]
            dataset[index] = iteration_results
        # params = np.tile(self.sim.get_parameters(), (self.n_iter, 1))

        # concat into one big dataset and scale
        tmp = []
        tmp_y = []
        tmp_s = []

        for set in dataset:
            # cc = np.concatenate((csv_inputs, params, set['x'], set['s']), axis=1)
            cc = np.concatenate((self.csv_inputs, set['x'], set['s']), axis=1)
            tmp_y.append(set['y'])
            tmp_s.append(set['s'])
            tmp.append(cc)

        in_values = np.concatenate(tmp, axis=0)[1:]
        concat_y = np.concatenate(tmp_y, axis=0)
        concat_s = np.concatenate(tmp_s, axis=0)
        delta_y = concat_y[1:] - concat_y[:-1]
        delta_s = concat_s[1:] - concat_s[:-1]
        concat_gpr_out = np.concatenate((delta_y, delta_s), axis=1)
        out_values = concat_gpr_out
        #
        # if self.scale:
        #     print('scale')
        #     self.x_scaler = MinMaxScaler()
        #     in_values = self.x_scaler.fit_transform(in_values)
        #     self.y_scaler = MinMaxScaler()
        #     scaled_out = self.y_scaler.fit_transform(concat_gpr_out)
        #     out_values = scaled_out
        #
        # if self.reduce_dims:
        #     print('reducing dims')
        #     # use PCA to reduce dimensionality
        #     i = 1
        #     ratio = [0]
        #     while i < (self.len_u + self.len_s + self.len_x) and sum(ratio) < 0.95:
        #     # while i < (self.len_u + self.len_s + self.len_x + params.shape[1]) and sum(ratio) < 0.95:
        #         self.pca = PCA(n_components=i)
        #         self.pca.fit(in_values)
        #         ratio = self.pca.explained_variance_ratio_
        #         i += 1
        #     return self.pca.transform(in_values), out_values
        return in_values, out_values

    def get_initial_state(self, params: np.ndarray = None):
        variable_filter = self.sim.set_variable_filter([self.framework.state_variables,
                                                        self.framework.objective_variables])
        variables = ' -override ' + variable_filter
        if params is not None:
            x_value = self.framework.control_variables.get_values_from_indices(params)[0]
            variables += ','
            for i, var in enumerate(self.framework.control_variables):
                variables += var.name + '=' + str(x_value[i]) + ','
            self.sim.write_inputs(x_value)

        result = self.sim.execute_model(
            variables=variables,
            start_time=self.sim.start_time,
            stop_time=self.sim.step_size + self.sim.start_time)
        solutions = result[-2, 1:]
        y = solutions[:self.len_y]
        y = (y + self.sim.offset) * self.sim.scale
        s = solutions[self.len_y:self.len_s + 1]

        if self.log:
            self.database['x'][0] = np.zeros(self.len_x)
            self.database['s'][0] = s
            self.database['y'][0] = y
        return y, s