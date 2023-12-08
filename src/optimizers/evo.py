from base import optimizer
from search_based import random_search
import numpy as np
import random as rd
from deap import base, creator, tools
from scipy.optimize import differential_evolution


class evolutionary(optimizer):
    def __init__(self, strategy: str = 'best1bin', init: str = 'latinhypercube', tolerance: float = 0.01,
                 size_individual: int = None, size_population: int = 300, bitsize_per_variable: np.ndarray = None,
                 tournSel_k: int = 10, MUTPB: float | tuple[float, float] = None, XOPB: float = 0.9,
                 *args, **kwargs):

        try:
            if kwargs['use_scipy']:
                self.strategy = strategy
                self.init = init
                self.tolerance = tolerance
                self.size_population = size_population
                self.MUTBP = (0.5, 1) if MUTPB is None else MUTPB
                self.XOPB = XOPB
        except KeyError:
            kwargs.update({'use_scipy': False})
        if not kwargs['use_scipy']:
            self.size_population = size_population
            self.size_individual = size_individual
            self.bitsize_per_variable = bitsize_per_variable

            # XOPB  is the probability with which two individuals
            #       are crossed or mated
            # MUTPB is the probability for mutating an individual
            self.XOPB = XOPB
            self.MUTPB = (0.5, 1) if MUTPB is None else MUTPB

            # variable for tournament selection
            self.tournSel_k = tournSel_k

            # create Fitness func and individual
            creator.create('FitnessMin', base.Fitness, weights=(-1,))
            creator.create('Individual', list, fitness=creator.FitnessMin)

            # instance of Toolbox class - define necessary func's after framework has been set during run()
            self.toolbox = base.Toolbox()

            ### STATS
            self.stats = tools.Statistics()
            self.stats.register('Min', np.min)
            self.stats.register('Max', np.max)
            self.stats.register('Avg', np.mean)
            self.stats.register('Std', np.std)

            self.logbook = tools.Logbook()

            ### HALL OF FAME (stores the best performing individuals)
            self.hall_of_fame = tools.HallOfFame(1)
        super().__init__(*args, **kwargs)

    def set_framework(self, framework):
        super().set_framework(framework)

        if not self.scipy:
            # set size of individual
            if self.size_individual is None:
                if self.bitsize_per_variable is None:
                    # determine necessary bitsize from transformed decision variables of framework
                    sample_space_lenghts = [len(s) for s in self.framework.decision_variables.get_sample_space()]
                    scalings = self.framework.decision_variables.get_scaling()

                    def bit_lenght(n):
                        bits = 0
                        while n >> bits: bits += 1
                        return bits

                    # self.bitsize_per_variable = np.array([bit_lenght(l) if scalings[i] == 'linear'
                    #                                       else l for i, l in enumerate(sample_space_lenghts)])
                    self.bitsize_per_variable = np.array([bit_lenght(l) for l in sample_space_lenghts])
                    self.size_individual = np.sum(self.bitsize_per_variable)
                else:
                    self.size_individual = np.sum(self.bitsize_per_variable)

            # create attribute constructor
            self.toolbox.register('attr_bool', rd.randint, 0, 1)
            # create individual constructor
            self.toolbox.register('individual', tools.initRepeat, creator.Individual, self.toolbox.attr_bool,
                                  self.size_individual)
            # create population constructor
            self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)

            ### EXPLANATION:
            # when toolbox.individual is called, an individual the size of self.size_individual containing 0's and 1's
            # is created.
            # toolbox.population will then create a list the size of self.size_population filled with individual

            # define objective function
            self.toolbox.register("evaluate", self.step)

            # set parallel evaluation mapping
            if self.parallel:
                self.toolbox.register("map", self.parallel_evaluator.init_pool().map)

            # set constraints
            avg_x = (self.framework.decision_variables.get_upper_bound()
                     - self.framework.decision_variables.get_lower_bound()) / 2
            #penalty_value = self.model.predict(avg_x)[0]
            #distance_penalty = lambda x_iter: self.framework.constraint_penalty(self.decode_x(x_iter), penalty_value)
            self.toolbox.decorate("evaluate", tools.DeltaPenalty(self.check_constraints,
                                                                 0))#penalty_value * 10,
                                                                 #distance_penalty))

            # define mating and mutating process using DEAP library
            self.toolbox.register("mate", tools.cxTwoPoint)  # XO strategy
            self.toolbox.register("mutate", tools.mutFlipBit,
                                  indpb=0.05)  # mutation strategy
            self.toolbox.register("select", tools.selTournament, tournsize=self.tournSel_k)  # selection strategy

    def decode_x(self, individual):
        # converts binary variables in chromosomes to decimal values

        # get size information
        if self.bitsize_per_variable is None:
            len_var_in_chromosome = int(self.size_individual / self.len_x)
            self.bitsize_per_variable = np.full(self.len_x, len_var_in_chromosome)

        # create result list
        out = np.zeros(self.len_x, dtype=float)

        # loop through every bit-sequence of variables in individual (e.g. 81bit, 3 vars --> 27bit per var)
        index = 0
        for i, size_var_in_chromosome in enumerate(self.bitsize_per_variable):
            # if self.framework.decision_variables.get_scaling(i) == 'linear':
            # convert binary to decimal (2**place_value method) --> this is some black magic fuckery
            chromosome_string = ''.join((str(j) for j in individual[index:index + size_var_in_chromosome - 1]))
            binary_to_decimal = int(chromosome_string, 2)
            # get precision of conversion
            precision = (self.ub[i] - self.lb[i]) / ((2 ** size_var_in_chromosome) - 1)
            # store decoded value
            out[i] = (binary_to_decimal * precision) + self.lb[i]
            # else:
            #     # reverse one hot encoding
            #     chromosome_string = ''.join((str(j) + ' ' for j in individual[index:index + size_var_in_chromosome - 1]))
            #     one_hot = np.fromstring(chromosome_string, dtype=int, sep=' ')
            #     out[i] = np.argmax(one_hot)
            index += size_var_in_chromosome
        transformed_out = self.framework.round_indices(out)
        return transformed_out

    def step(self, individual):
        # see how individuals perform
        x = self.decode_x(individual)
        return self.model.predict(x)

    def check_constraints(self, individual):
        # check if variable constrains are followed.
        # Return True if okay, False if constraint(s) breached
        var_list = self.decode_x(individual)
        return self.framework.check_constraints(var_list)

    def penalty(self, individual):
        # penalty function if constraints are breached
        var_list = self.decode_x(individual)
        return np.sum(var_list) ** 2

    def run(self):
        # create initial population and evaluate its fitness
        population = self.toolbox.population(n=self.size_population)
        fitness_list = list((map(self.toolbox.evaluate, population)))

        for individual, fitness in zip(population, fitness_list):
            individual.fitness.values = fitness

        for gen in range(self.n_iter):
            # create new generation
            offspring = self.toolbox.select(population, self.size_population)
            # clone the offspring (important because following modification will happen in place)
            offspring = list((map(self.toolbox.clone, offspring)))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if rd.random() < self.XOPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                MUTPB_gen = rd.uniform(self.MUTPB[0], self.MUTPB[1]) \
                                    if isinstance(self.MUTPB, tuple) else self.MUTPB

                if rd.random() < MUTPB_gen:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate fitness of invalid individuals
            invalid_individual = [individual for individual in offspring if not individual.fitness.valid]
            fitness_list = list((map(self.toolbox.evaluate, invalid_individual)))
            for individual, fitness in zip(invalid_individual, fitness_list):
                individual.fitness.values = fitness

            # replace population by offspring
            population[:] = offspring

            # Evaluate performance of generation by compiling min, max, mean and std deviation
            gen_performance = [individual.fitness.values[0] for individual in population]

            # Store the best individual of the current generation
            if self.log:
                x = tools.selBest(population, k=1)[0]
                self.database['x'][gen] = self.decode_x(x)
                self.database['y'][gen] = x.fitness.values

            self.hall_of_fame.update(offspring)
            gen_stats = self.stats.compile(gen_performance)

            gen_stats['Generation'] = gen  # create key
            self.logbook.append(gen_stats)

            # print(gen_stats)

        best_individual = self.hall_of_fame[0]
        opt_x = self.decode_x(best_individual)
        opt_y = best_individual.fitness.values
        return opt_x, opt_y

    def run_scipy(self) -> ([np.ndarray.dtype, np.ndarray.dtype], np.ndarray.dtype):
        random_x = random_search(iterations=1, objective_model=self.model, framework=self.framework).generate_x()
        opt = differential_evolution(
            func=self.model.predict,
            bounds=self.bounds,
            x0=random_x,
            strategy=self.strategy,
            maxiter=self.n_iter,
            popsize=self.size_population,
            tol=self.tolerance,
            mutation=self.MUTBP,
            recombination=self.XOPB,
            polish=False,
            init=self.init,
            constraints=self.framework.constraints,
            workers=self.n_workers
        )

        opt_x = opt.x
        opt_y = opt.fun

        return opt_x, opt_y