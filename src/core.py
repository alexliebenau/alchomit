import math
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import NonlinearConstraint

# enable eager execution of TF
tf.config.run_functions_eagerly(
    run_eagerly=True
)

VAL_VARIABLE_TYPE = (float | int | np.number)
VAL_ARRAY_TYPE = (list[float] | list[int] | np.ndarray[float] | np.ndarray[int] | tf.Tensor)
VAL_TYPE = VAL_VARIABLE_TYPE | VAL_ARRAY_TYPE
IDENTIFIER_TYPE = (int | str | np.int_ | np.ndarray[str] | np.ndarray[int] | list[str] | list[int] | type(None))
INDEX_VALUE_ARRAY_TYPE = (np.ndarray[int] | list[int] | tf.Tensor)


class variable:
    def __init__(self,
                 name: str,
                 lb: float | int,
                 ub: float | int,
                 val_type: str = 'real',  # real, discrete, switch, multiple_switch
                 acc: float = None,
                 scaling: str = None  # linear, exp2
                 ):

        self.name = name
        self.lower_bound = lb
        self.upper_bound = ub
        self.accuracy = acc
        self.scaling = scaling
        self.val_type = val_type
        self.sample_space = self.build_sample_space()
        # self.stored_values = np.empty(0, dtype=float)
        # match val_type:
        #     case 'real':
        #         self.stored_values = np.empty(0, dtype=float)
        #     case _:
        #         self.stored_values = np.empty(0, dtype=int)

    def __str__(self) -> str:
        return ''.join(f"    ------------  Variable '{self.name}':\n" +
                       f"     Lower Bound: '{self.lower_bound}'\n" +
                       f"     Upper Bound: '{self.upper_bound}'\n" +
                       f"        Accuracy: '{self.accuracy}'\n" +
                       f"         Scaling: '{self.scaling}'\n" +
                       f"      Value type: '{self.val_type}'\n" +
                       f"Dim Sample Space: '{'continuous' if self.sample_space is None else len(self.sample_space)}'\n"
                       # f" Dim Stored Vals: '{len(self.stored_values)}'\n"
                       )

    def __repr__(self) -> str:
        return 'Variable(' + self.name + ')'

    def get_transformed_sample_space(self):
        lower_bound_t = self.transform(self.lower_bound)
        upper_bound_t = self.transform(self.upper_bound)
        n_samples = upper_bound_t - lower_bound_t + 1
        sample_space_t = np.linspace(start=lower_bound_t,
                                     stop=upper_bound_t,
                                     num=n_samples,
                                     dtype=np.float64)
        return self.inverse_transform(sample_space_t)

    def build_sample_space(self) -> np.ndarray[np.float64 | np.int8]:
        match self.val_type:
            case "real":
                # create sample space consisting of 100 elements for parameter grid
                sample_space = np.linspace(start=self.lower_bound,
                                           stop=self.upper_bound,
                                           num=100,
                                           dtype=np.float64)
                return sample_space
            case "discrete":
                return self.get_transformed_sample_space()
            # case "switch":
            #     # build sample space with one-hot encoding
            #     enumerated_values = np.array((range(self.lower_bound, self.upper_bound)))
            #     one_hot = np.zeros((enumerated_values.size, enumerated_values.max() + 1), dtype=np.int8)
            #     one_hot[np.arange(enumerated_values.size), enumerated_values] = 1
            #     return one_hot
            # case "multiple_switch":
            #     # build sample space of all possible switch positions
            #     dim = self.upper_bound - self.lower_bound
            #     return np.array(np.meshgrid(*([0, 1],) * dim), dtype=np.int8).T.reshape(-1, dim)
            case _:
                raise TypeError('Unknown variable type specified: ', self.val_type)

    def transform(self, val: VAL_TYPE) -> VAL_TYPE:
        # match for val, np and tensor
        if self.val_type == 'real':
            val_t = val
        else:
            if isinstance(val, float | int | np.number):
                match self.scaling:
                    case 'linear':
                        val_t = int(round(val / self.accuracy))
                    case 'exp2':
                        val_t = int(round(math.log(val, self.accuracy)))
                    case _:
                        raise ValueError("This scaling type has not been implemented: ", print(self.scaling))
            elif isinstance(val, np.ndarray):
                match self.scaling:
                    case 'linear':
                        val_t = np.round((val / self.accuracy))
                    case 'exp2':
                        val_t = np.round(np.log2(val))
                    case _:
                        raise ValueError("This scaling type has not been implemented: ", print(self.scaling))
            elif isinstance(val, tf.Tensor):
                match self.scaling:
                    case 'linear':
                        val_t = tf.math.round((val / self.accuracy))
                    case 'exp2':
                        val_t = tf.math.round(tf.math.log(val) * 1.44)  # equals ln(val) / ln(2) = log2(val)
                    case _:
                        raise ValueError("This scaling type has not been implemented: ", print(self.scaling))
            else:
                raise TypeError("This conversion type has not been implemented: ", type(val))
        return val_t

    def inverse_transform(self, val_t: VAL_TYPE) -> VAL_TYPE:
        # match for val, np and tensor
        if isinstance(val_t, float | int):
            match self.scaling:
                case 'real':
                    val = val_t
                case 'linear':
                    val = val_t * self.accuracy
                case 'exp2':
                    val = 2 ** val_t
                case _:
                    raise ValueError("This scaling type has not been implemented: ", print(self.scaling))
        elif isinstance(val_t, np.ndarray):
            match self.scaling:
                case 'real':
                    val = val_t
                case 'linear':
                    val = val_t.astype(float) * self.accuracy
                case 'exp2':
                    val = 2 ** val_t
                case _:
                    raise ValueError("This scaling type has not been implemented: ", print(self.scaling))
        else:
            raise TypeError("This conversion type has not been implemented: ", type(val_t))
        return val

    # def store_value(self, val: VAL_TYPE):
    #     if isinstance(val, VAL_VARIABLE_TYPE):
    #         val = np.array([val])
    #     elif isinstance(val, list):
    #         val = np.array(val)
    #
    #     res = np.concatenate((self.stored_values, val), axis=0)
    #     self.stored_values = res


VAR_ARRAY_INPUT_ARRAY_TYPE = (list[variable] | np.ndarray[variable])
VAR_ARRAY_INPUT_TYPE = (variable | VAR_ARRAY_INPUT_ARRAY_TYPE)


class variable_array:
    def __init__(self, input_vars: VAR_ARRAY_INPUT_TYPE = None):
        self.variables = np.empty(0, dtype=variable)
        if input_vars is not None:
            self.add_variable(input_vars)

        # property callers
    def get_name(self, i=None): return self.get_property(i, 'name')

    def get_lower_bound(self, i=None): return self.get_property(i, 'lower_bound')

    def get_upper_bound(self, i=None): return self.get_property(i, 'upper_bound')

    def get_scaling(self, i=None): return self.get_property(i, 'scaling')

    def get_accuracy(self, i=None): return self.get_property(i, 'accuracy')

    def get_val_type(self, i=None): return self.get_property(i, 'val_type')

    def get_sample_space(self, i=None): return self.get_property(i, 'sample_space')

    def __len__(self):
        return len(self.variables)

    def __getitem__(self, index):
        return self.variables[index]

    def __setitem__(self, index, new_var):
        if isinstance(new_var, variable):
            self.variables[index] = new_var
        else:
            raise TypeError("Variable has to be of type variable() to be inserted, got type ", type(new_var))

    def __str__(self):
        string = ''
        for var in self.variables:
            string += (var.__str__() + '\n ----------------------------------------- \n')
        return string

    def __repr__(self):
        if len(self) > 1:
            string = ''
            for var in self.variables[:-1]:
                string += var.name + ', '
            string += self.variables[-1]
        elif len(self) == 1:
            string = self.variables[0].name
        elif len(self) == 0:
            return 'Empty variable array'
        return f"Variable array containing '{string}'"

    def add_variable(self, var: VAR_ARRAY_INPUT_TYPE):
        if isinstance(var, list) | isinstance(var, np.ndarray):
            for v in var:
                self.add_variable(v)
        elif isinstance(var, variable):
            self.variables = np.concatenate((self.variables, np.array([var])))
        else:
            raise TypeError('Unknown type to add to variable array: ', type(var))

    def get_property(self, identifier: IDENTIFIER_TYPE, property_name: str):
        if identifier is None:
            return np.array([self.variables[i].__getattribute__(property_name) for i in range(len(self.variables))])
        elif isinstance(identifier, int):
            return self.variables[identifier].__getattribute__(property_name)
        elif isinstance(identifier, np.ndarray | list):
            return np.array([self.variables[i].__getattribute__(property_name) for i in identifier])
        elif isinstance(identifier, str):
            index = np.where(self.get_name() == identifier)[0][0]
            return self.variables[index].__getattribute__(property_name)
        else:
            raise TypeError('Unknown type for identifier: ', type(identifier))

    def get_values_from_indices(self, index_array: INDEX_VALUE_ARRAY_TYPE) -> INDEX_VALUE_ARRAY_TYPE:
        is_single_value = False
        if isinstance(index_array, list):
            index_array = np.array(index_array)
        type_to_array = (float | int | np.number)
        if isinstance(index_array, type_to_array):
            is_single_value = True
            index_array = np.array([index_array])
        if len(index_array.shape) > 1:
            out = []
            for arr in index_array:
                out.append(self.get_values_from_indices(arr))
            return out

        values = np.zeros(index_array.shape, dtype=float)
        for i, index in enumerate(index_array):
            if self.variables[i].val_type == 'real':
                values[i] = index
            else:
                values[i] = self.variables[i].sample_space[int(index)]

        if is_single_value:
            values = values[0]
        if isinstance(index_array, np.ndarray):
            return values
        elif isinstance(index_array, tf.Tensor):
            return tf.convert_to_tensor(values)
        else:
            raise TypeError("Unknown type for index array: ", type(index_array))

    def get_indices_from_values(self, value_array: INDEX_VALUE_ARRAY_TYPE) -> INDEX_VALUE_ARRAY_TYPE:
        is_single_value = False
        if isinstance(value_array, list):
            value_array = np.array(value_array)
        type_to_array = (float | int | np.number)
        if isinstance(value_array, type_to_array):
            is_single_value = True
            value_array = np.array([value_array])
        indices = np.zeros(value_array.shape)
        if isinstance(value_array, float) or isinstance(value_array, int):
            value_array = np.array(value_array)

        for i, value in enumerate(value_array):
            match self.variables[i].val_type:
                case 'real':
                    indices[i] = value
                case 'discrete':
                    rounded_value = self.variables[i].transform(value)
                    indices[i] = (np.abs(value_array - value)).argmin()
                # case 'switch' | 'multiple_switch':
                #     indices[i] = np.where(self.variables[i].sample_space == value)[0][0]
        if is_single_value:
            indices = indices[0]
        if isinstance(value_array, np.ndarray):
            return indices
        elif isinstance(value_array, tf.Tensor):
            return tf.convert_to_tensor(indices)
        else:
            raise TypeError("Unknown type for value array: ", type(value_array))


class nonlinear_constraint(NonlinearConstraint):
    def __init__(self, fun, lb=float('-inf'), ub=float('-inf')):
        super().__init__(fun, lb, ub)
        self.var_array = None

    def transformed_constraint(self, var_array: variable_array):
        self.var_array = var_array
        return nonlinear_constraint(
            fun=self.transformed_fun,
            lb=self.lb,
            ub=self.ub
        )

    def transformed_fun(self, X):
        X_d = self.var_array.get_values_from_indices(X)
        return self.fun(X_d)


class linear_constraint(nonlinear_constraint):
    def __init__(self, A, lb=float('-inf'), ub=float('inf')):
        if isinstance(A, list):
            A = np.array(A)
        self.A = A
        super().__init__(fun=self.fun, lb=lb, ub=ub)

    def fun(self, x): return self.A @ x


CONSTRAINT_VARIABLE_TYPE = (linear_constraint | nonlinear_constraint)
CONSTRAINT_ARRAY_TYPE = (list[CONSTRAINT_VARIABLE_TYPE] | np.ndarray[CONSTRAINT_VARIABLE_TYPE])
CONSTRAINT_TYPE = CONSTRAINT_VARIABLE_TYPE | CONSTRAINT_ARRAY_TYPE


class sub_framework:
    def __init__(self,
                 decision_variables: VAR_ARRAY_INPUT_TYPE | variable_array,
                 objective_variables: VAR_ARRAY_INPUT_TYPE | variable_array,
                 state_variables: VAR_ARRAY_INPUT_TYPE | variable_array = None,
                 constraints: CONSTRAINT_TYPE = None
                 ):

        if isinstance(decision_variables, variable_array):
            self.decision_variables = decision_variables
        else:
            self.decision_variables = variable_array(decision_variables)

        self.objective_variables = objective_variables

        if isinstance(state_variables, variable_array):
            self.state_variables = state_variables
        else:
            self.state_variables = variable_array(state_variables)

        self.constraints = constraints

    def __str__(self):
        string = ''

        def add_to_string(var_arr: variable_array, name: str, string: str):
            if len(var_arr.variables) > 0:
                string += ('\n   {} VARIABLES     ---------------------------  \n'.format(name.upper()))
                string += var_arr.__str__()
            else:
                string += ('\n   {} VARIABLES     is empty     --------------- \n'.format(name.upper()))
            return string

        string = add_to_string(self.decision_variables, 'decision', string)
        # string = add_to_string(self.objective_variables, 'objective', string)
        return string

    def __repr__(self):
        return f"Framework containing: \n" + \
               f"'{len(self.objective_variables)}' objective variables \n"

    def round_indices(self, arr: INDEX_VALUE_ARRAY_TYPE) -> INDEX_VALUE_ARRAY_TYPE:
        if isinstance(arr, np.ndarray) | isinstance(arr, tf.Tensor):
            # arr_t = np.array((arr.shape), dtype=np.float64)
            # dim = 1 if len(arr.shape) == 1 else arr.shape[0]
            dim = len(arr.shape)
        elif isinstance(arr, list):
            # arr_t = arr
            # dim = 2 if isinstance(arr[0], list) else 1
            return self.round_indices(np.array(arr))
        else:
            raise TypeError('Unknown type of index array {}'.format(type(arr)))
        if dim > 1:
            if isinstance(arr, np.ndarray) | isinstance(arr, list):
                arr_t = arr
                for i, a in enumerate(arr):
                    arr_t[i] = self.round_indices(a)

            elif isinstance(arr, tf.Tensor):
                def transform_fxn(x: VAL_VARIABLE_TYPE, var: variable):
                    return var.transform(x)

                vectorized_transform = np.vectorize(transform_fxn, otypes=[float])
                arr_t = tf.convert_to_tensor(vectorized_transform(arr, np.array([self.decision_variables])))
                return arr_t

            else:
                raise TypeError(f'Unknown type {type(arr)} as input array for round_indices')

            return arr_t

        else:
            if len(arr.shape) > 1:
                arr = arr[0]
            arr_t = np.zeros(arr.shape)
            if isinstance(arr, list) | isinstance(arr, np.ndarray):
                for i, var in enumerate(self.decision_variables):
                    arr_t[i] = var.transform(arr[i])
            elif isinstance(arr, tf.Tensor):
                arr_t = tf.convert_to_tensor(arr_t)
                mask = np.zeros(arr_t.shape)
                for i, var in enumerate(self.decision_variables):
                    mask[i] = 1
                    arr_t += arr_t * 0 + mask * var.transform(arr[i])
                    mask[i] = 0

            return arr_t

    def set_simulation_state_variables(self, sim):
        state_vars = sim.get_sim_variables()['state_vars']
        for name in state_vars:
            var = variable(name=name, lb=-float('inf'), ub=float('inf'), val_type='real')
            self.state_variables.add_variable(var)

    def check_constraints(self, x) -> bool:
        if self.constraints is None:
            return True
        rounded_x = self.round_indices(x)
        for constraint in self.constraints:
            if not constraint.lb <= constraint.fun(rounded_x) <= constraint.ub:
                return False
        return True

    def constraint_penalty(self, x, random_value) -> VAL_TYPE:
        rounded_x = self.round_indices(x)
        penalty_value = 0

        def quadratic_distance_penalty(var, val1, val2):
            if (val1 - val2) < 1:
                return random_value * ((val1 - val2) ** 2)
            elif (val1 - val2) > 1:
                return random_value / ((val1 - val2) ** 2)
            elif var == 0:
                return 0

        for constraint in self.constraints:
            if constraint.fun(rounded_x) < constraint.lb:
                penalty_value += quadratic_distance_penalty(random_value, constraint.fun(rounded_x), constraint.lb)
            elif constraint.fun(rounded_x) > constraint.ub:
                penalty_value += quadratic_distance_penalty(random_value, constraint.fun(rounded_x), constraint.ub)
        return penalty_value


class framework(sub_framework):
    def __init__(self,
                 objective_variables: VAR_ARRAY_INPUT_TYPE,
                 optimisation_variables: VAR_ARRAY_INPUT_TYPE = None,
                 control_variables: VAR_ARRAY_INPUT_TYPE = None,
                 state_variables: VAR_ARRAY_INPUT_TYPE = None,
                 optimisation_constraints: CONSTRAINT_TYPE | CONSTRAINT_ARRAY_TYPE = None,
                 control_constraints: CONSTRAINT_TYPE | CONSTRAINT_ARRAY_TYPE = None
                 ):

        self.optimisation_variables = variable_array(optimisation_variables)
        self.control_variables = variable_array(control_variables)
        self.state_variables = variable_array(state_variables)
        self.objective_variables = objective_variables

        # transform to discrete space
        def get_transformed_vars(var_array: variable_array) -> variable_array:
            trans_var_array = variable_array()
            for var in var_array.variables:
                match var.val_type:
                    case 'real':
                        trans_var_array.add_variable(var)
                    case _:
                        var = variable(name=var.name, lb=0, ub=len(var.sample_space) - 1,
                                       scaling='linear', acc=1, val_type='discrete')
                        trans_var_array.add_variable(var)
            return trans_var_array

        self.transformed_optimisation_variables = get_transformed_vars(self.optimisation_variables)
        self.transformed_control_variables = get_transformed_vars(self.control_variables)
        self.transformed_state_variables = get_transformed_vars(self.state_variables)
        # self.transformed_objective_variables = get_transformed_vars(self.objective_variables)

        # concat opt and contr vars into decision vars
        decision_variables = np.concatenate((self.transformed_optimisation_variables.variables,
                                             self.transformed_control_variables.variables))

        super().__init__(decision_variables=decision_variables,
                         objective_variables=self.objective_variables,
                         state_variables=self.state_variables)

        self.optimisation_constraints = np.empty(0, dtype=NonlinearConstraint)
        if optimisation_constraints is not None:
            for constraint in optimisation_constraints:
                self.add_optimisation_constraint(constraint)

        self.control_constraints = np.empty(0, dtype=NonlinearConstraint)
        if control_constraints is not None:
            for constraint in control_constraints:
                self.add_control_constraint(constraint)

    def add_optimisation_constraint(self, constraint: CONSTRAINT_TYPE):
        self.optimisation_constraints = self.add_constraint(constraint, self.optimisation_constraints)

    def add_control_constraint(self, constraint: CONSTRAINT_TYPE):
        self.control_constraints = self.add_constraint(constraint, self.control_constraints)

    def add_constraint(self, constraint: CONSTRAINT_TYPE, array):
        if isinstance(constraint, list) | isinstance(constraint, np.ndarray):
            for c in constraint:
                array = self.add_constraint(c, array)
            return array
        elif isinstance(constraint, nonlinear_constraint) | isinstance(constraint, linear_constraint):
            return np.concatenate((array, np.array([constraint])))
        else:
            raise TypeError('Unknown constraint type: {}'.format(type(constraint)))

    def get_optimisation_subframework(self):
        return sub_framework(
            decision_variables=self.transformed_optimisation_variables,
            objective_variables=self.objective_variables,
            constraints=[cnstr.transformed_constraint(self.optimisation_variables)
                         for cnstr in self.optimisation_constraints] if self.optimisation_constraints is not None
            else None
        )

    def get_control_subframework(self):
        # remember to add reduced state variables!:
        # for var in self.control_variables:
        #     if var.val_type == 'real':
        #         raise ValueError(f'Continuous variables not supported for control. '
        #                          f'Please discretize control variables.')
        return sub_framework(
            decision_variables=self.transformed_control_variables,
            objective_variables=self.objective_variables,
            state_variables=self.transformed_state_variables,
            constraints=[cnstr.transformed_constraint(self.control_variables)
                         for cnstr in self.control_constraints] if self.control_constraints is not None
            else None
        )

    def log(self, module, var_array: variable_array):
        res_vars = var_array.get_name()
        header = np.concatenate((res_vars, self.objective_variables))
        data_x = module.database['x']
        data_y = module.database['y']
        for i, val in enumerate(data_x):
            data_x[i] = var_array.get_values_from_indices(val)
        data = np.concatenate((data_x, data_y), axis=1)
        dataframe = pd.DataFrame(data=data, columns=header)
        print(dataframe.to_string(index=True))
        dataframe.to_csv('optimisation.csv', index=True)

    def optimize(self, optimizer):
        if optimizer.framework is None:
            opt_fw = self.get_optimisation_subframework()
            optimizer.set_framework(opt_fw)
            if len(self.optimisation_constraints) > 0:
                optimizer.constrained = True

        x_t, y = optimizer.minimize()
        x = self.optimisation_variables.get_values_from_indices(x_t)

        if optimizer.log:
            self.log(optimizer, self.optimisation_variables)
        return x, y

    def control(self, controller):
        # self.set_simulation_state_variables(controller.sim)
        if controller.framework is None:
            ctrl_fw = self.get_control_subframework()
            controller.set_framework(ctrl_fw)

        y = controller.predict()
        # x = self.control_variables.get_values_from_indices(x_t)

        if controller.log:
            # self.log(controller, self.control_variables)
            print(f'Result: {y}')
            x = controller.stored_x
            print(f'Best ctrl values: {controller.sim.write_inputs(x)}')
            actual_y = controller.sim.run()
            diff = y - actual_y
            print(f'Difference between prediction and simulation result: {diff}')
        return y

    def dynamic_optimize(self, optimizer, controller):
        if controller.framework is None:
            ctrl_fw = self.get_control_subframework()
            controller.set_framework(ctrl_fw)

        if optimizer.framework is None:
            opt_fw = self.get_optimisation_subframework()
            optimizer.set_framework(opt_fw)

        optimizer.model = controller
        x_t, y = optimizer.minimize()
        x = self.optimisation_variables.get_values_from_indices(x_t)

        if optimizer.log:
            self.log(optimizer, self.optimisation_variables)
        return x, y
