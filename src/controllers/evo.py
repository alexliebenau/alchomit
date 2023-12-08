from base import controller
from ..optimizers import evolutionary
import numpy as np

class genetic_ctrl(controller):
    def __init__(self, iterations: int, size_individual: int = None, size_population: int = 300,
                 bitsize_per_variable: np.ndarray = None,
                 tournSel_k: int = 10, MUTPB: float | tuple[float, float] = None, XOPB: float = 0.9,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen = None
        self.iterations = iterations
        self.size_individual = size_individual
        self.size_population = size_population
        self.bitsize_per_variable = bitsize_per_variable
        self.tournSel_k = tournSel_k
        self.MUTPB = MUTPB
        self.XOPB = XOPB

        self.bitsize_per_variable = None
        self.params = None

    def set_framework(self, framework):
        super().set_framework(framework)
        self.get_instance()

    def get_instance(self):
        self.gen = evolutionary(
            iterations=self.iterations,
            objective_model=self.function_approximator,
            size_individual=self.size_individual,
            size_population=self.size_population,
            bitsize_per_variable=self.bitsize_per_variable,
            tournSel_k=self.tournSel_k,
            MUTPB=self.MUTPB,
            XOPB=self.XOPB,
            use_scipy=False,
            parallel=self.parallel,
            log=False
        )
        # multiply bitlenght with iterations
        sample_space_lenghts = [len(s) for s in self.framework.decision_variables.get_sample_space()]
        scalings = self.framework.decision_variables.get_scaling()

        def bit_lenght(n):
            bits = 0
            while n >> bits: bits += 1
            return bits

        # self.bitsize_per_variable = np.array([bit_lenght(l) if scalings[i] == 'linear'
        #                                       else l for i, l in enumerate(sample_space_lenghts)])
        self.bitsize_per_variable = np.array([bit_lenght(l) for l in sample_space_lenghts])
        bitsize_per_variable = np.array([bit_lenght(l) * self.n_iter for l in sample_space_lenghts])
        self.gen.bitsize_per_variable = bitsize_per_variable
        self.gen.size_individual = np.sum(bitsize_per_variable)
        # set framework for gen
        self.gen.set_framework(self.framework)
        self.gen.decode_x = self.decode_x
        self.gen.check_constraints = lambda fuckoff: True
        self.gen.step = self.gen_step
        self.gen.toolbox.register("evaluate", self.gen_step)



    def decode_x(self, individual):
        # converts binary variables in chromosomes to decimal values

        # create result list
        out = np.zeros((self.n_iter, self.len_x), dtype=float)
        # loop through every bit-sequence of variables in individual (e.g. 81bit, 3 vars --> 27bit per var)
        index = 0
        for i, size_var_in_chromosome in enumerate(self.bitsize_per_variable):
            for iter in range(self.n_iter):
                # if self.framework.decision_variables.get_scaling(i) == 'linear':
                # convert binary to decimal (2**place_value method) --> this is some black magic fuckery
                chromosome_string = ''.join((str(j) for j in individual[index:index + size_var_in_chromosome - 1]))
                binary_to_decimal = int(chromosome_string, 2)
                # get precision of conversion
                precision = (self.ub[i] - self.lb[i]) / ((2 ** size_var_in_chromosome) - 1)
                # store decoded value
                out[iter, i] = (binary_to_decimal * precision) + self.lb[i]

            # else:
            #     # reverse one hot encoding
            #     chromosome_string = ''.join((str(j) + ' ' for j in individual[index:index + size_var_in_chromosome - 1]))
            #     one_hot = np.fromstring(chromosome_string, dtype=int, sep=' ')
            #     out[i] = np.argmax(one_hot)
                index += size_var_in_chromosome
        rounded_out = np.zeros(out.shape, dtype=float)
        for i in range(out.shape[0]):
            rounded_out[i, :] = self.framework.round_indices(out[i, :])
        return rounded_out

    def gen_step(self, individual):
        x = self.decode_x(individual)
        for x_i in x:
            if not self.framework.check_constraints(x_i):
                print('failed')
                return np.array([0])
        print('passed')
        self.sim.set_inputs_from_list(x)
        variable_filter = self.sim.set_variable_filter(
            [self.framework.state_variables, self.framework.objective_variables])
        variables = ' -override ' + variable_filter
        result = (self.sim.execute_model(
            variables=variables)[:, 1:] + self.sim.offset) * self.sim.scale
        y = result[-1, :len(self.framework.objective_variables)]
        print(y)
        s = result[:, -len(self.framework.state_variables):]
        # # check constraints
        # print(result, result.shape)
        # for i in range(len(result)):
        #     if not self.framework.check_constraints(np.concatenate((x[i, :], s[i]))):
        #         return np.array([0])
        return y

    def predict(self, params=None) -> np.ndarray:
        self.params = params

        self.get_instance()

        opt_x, y = self.gen.minimize()
        self.stored_x = opt_x
        return y