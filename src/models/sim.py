from base import model
from ..core import framework as fw
from ..core import variable_array
import os
import subprocess
import glob
import numpy as np
import pandas as pd
import csv
import xml.dom.minidom as xmlparser
import xml.etree.ElementTree as xmlET
from tempfile import TemporaryDirectory
from OMPython import OMCSessionZMQ


class simulation(model):
    def __init__(self,
                 model_name: str,
                 package_path: str,
                 package_name: str,
                 stop_time: float | int,
                 step_size: int,
                 scale: float | int,
                 offset: float | int,
                 framework: fw.framework,
                 start_time: float | int = 0.0,
                 csv_path: str = None,
                 solver: str = 'dassl',
                 tolerance: float | int = 1e-06,
                 cflags: str = '',
                 simflags: str = '',
                 options: str = '',
                 libraries: str | list = None
                 ):

        self.model_name = model_name
        self.package_name = package_name
        self.start_time = start_time
        self.stop_stim = stop_time
        self.step_size = step_size
        self.iterations = round((stop_time - start_time) / step_size)
        self.scale = scale
        self.offset = offset
        self.framework = framework

        ### create temporary workspace
        self.dir = TemporaryDirectory()
        self.tmpdir = self.dir.name

        ### merge csv files into tmp folder
        self.csv_file_name = os.path.join(self.tmpdir, 'sim_input.csv')
        self.csv_input_file = pd.DataFrame()
        self.time_data = np.arange(start_time, stop_time, step_size)
        self.time_csv = pd.DataFrame(
            data=np.arange(start_time, stop_time, step_size),
            columns=['time']
        )

        if csv_path is not None:
            files = glob.glob(os.path.join(csv_path, '*.csv'))
            for file in files:
                f = pd.read_csv(file)
                for column in f:
                    if column.lower() != 'time' and 'unnamed' not in column.lower():
                        self.csv_input_file = pd.concat((self.csv_input_file, f.loc[:, column]), axis=1)

            self.csv_input_file = self.csv_input_file.loc[:, ~self.csv_input_file.columns.duplicated()]
        self.csv_input_file.to_csv(self.csv_file_name, sep=',', index=False)
        simflags += '-csvInput=' + self.csv_file_name

        # compile model in tmp folder
        session = OMCSessionZMQ()
        omc = lambda cmd: session.sendExpression(cmd)
        omc('cd("' + self.tmpdir + '")')
        omc('loadFile("' + package_path + '")')
        omc('loadModel(Modelica)')

        if libraries is not None:
            if isinstance(libraries, str):
                omc('loadModel(' + libraries + ')')
            elif isinstance(libraries, list) | isinstance(libraries, np.ndarray):
                for lib in libraries:
                    omc('loadModel(' + lib + ')')

        def get_variable_filter(framework: fw.framework):
            out = ''
            for var in framework.objective_variables:
                out += var + '|'
            return out[:-1]

        variables = get_variable_filter(framework)
        cmd = "buildModel(" + package_name + "." + model_name + ', ' \
              + 'startTime=' + str(start_time) + ', ' \
              + 'stopTime=' + str(stop_time) + ', ' \
              + 'numberOfIntervals=' + str(self.iterations) + ', ' \
              + 'tolerance=' + str(tolerance) + ', ' \
              + 'method="' + solver + '"' + ', ' \
              + 'fileNamePrefix="sim", ' \
              + 'options="' + options + '"' + ', ' \
              + 'outputFormat="csv", ' \
              + 'variableFilter="' + variables + '"' + ', ' \
              + 'cflags="' + cflags + '"' + ', ' \
              + 'simflags="' + simflags + '"' \
              + ")"

        compilation_result = omc(cmd)
        if compilation_result == ("", ""):
            raise RuntimeError('Compilation of model failed. Check compiler and simulation flags and the spelling of'
                               ' the objective variables! The execution command was:\n' + cmd)

        # make sure step_size in XML file is correct
        # (for whatever reason it seems to get that wrong sometimes - so here we ge for another dirty fix of things)
        xml_file = xmlET.parse(os.path.join(self.tmpdir, 'sim_init.xml'))
        settings = xml_file.findall('DefaultExperiment')[0]
        if settings.get('stepSize') != str(step_size):
            settings.set('stepSize', str(step_size))
            xml_file.write(os.path.join(self.tmpdir, 'sim_init.xml'))

        self.variable_array = None  # to be set from optimizer

    # Function to call Modelica simulation
    def execute_model(self,
                      variables: str | list,
                      step_size: float = '',
                      stop_time: float = '',
                      start_time: float = '',
                      flags: str = ''
                      ):
        if step_size != '':
            step_size = ',stepSize=' + str(step_size)
        if stop_time != '':
            stop_time = ',stopTime=' + str(stop_time)
        if start_time != '':
            if variables == '':
                start_time = ' -override startTime=' + str(start_time)
            else:
                start_time = ',startTime=' + str(start_time)

        simflags = ' -lv=-stdout,-LOG_SUCCESS' + flags
        hash_name = str(hash(variables)) + '.csv'
        exec_call = '(cd ' + self.tmpdir + ' && ' \
                                           './sim' + variables + start_time + stop_time + step_size \
                    + simflags + ' -r=' + hash_name + ')'
        subprocess.run(exec_call, shell=True)
        return np.genfromtxt(fname=os.path.join(self.tmpdir, hash_name), delimiter=',')

    def run(self, flags=''):
        result = self.execute_model(variables="", flags=flags)
        solutions = result[-1, 1:]
        y_value = (solutions + self.offset) * self.scale
        return y_value

    def predict(self, x_index):
        if x_index is not None:
            x_value = self.framework.optimisation_variables.get_values_from_indices(x_index)
            variables = ' -override '
            for i, var in enumerate(self.framework.optimisation_variables):
                variables += var.name + '=' + str(x_value[i]) + ','
        else:
            variables = ''

        # self.set_parameters(x_value)
        result = self.execute_model(variables=variables)
        solutions = result[-1, 1:]
        y_value = (solutions + self.offset) * self.scale
        return y_value

    def step(self, x_index, index):
        variable_filter = self.set_variable_filter([self.framework.state_variables, self.framework.objective_variables])
        variables = ' -override ' + variable_filter
        if x_index is not None:
            len_x = len(self.framework.control_variables)
            len_s = len(self.framework.state_variables)
            x_value = self.framework.control_variables.get_values_from_indices(x_index[:len_x])
            if len_s > 0:
                s_value = self.framework.state_variables.get_values_from_indices(x_index[-len_s:])
            variables += ','
            for i, var in enumerate(self.framework.control_variables):
                variables += var.name + '=' + str(x_value[i]) + ','
            for i, var in enumerate(self.framework.state_variables):
                variables += var.name + '=' + str(s_value[i]) + ','
            self.write_inputs(x_value.reshape(1, -1))

        result = self.execute_model(
            variables=variables,
            start_time=index * self.step_size + self.start_time,
            stop_time=(index + 1) * self.step_size + self.start_time)

        val2 = result[-2, 1:]
        val1 = result[1, 1:]
        delta_y = val2[:len(self.framework.objective_variables)] - val1[:len(self.framework.objective_variables)]
        delta_y = delta_y * self.scale
        delta_s = val2[-len(self.framework.state_variables):] - val1[-len(self.framework.state_variables):]
        return delta_y, delta_s

    def write_inputs(self, u: np.ndarray):  # change to set_inputs (to csv)
        input_tuple = {}
        # names = []

        for i in range(len(self.framework.control_variables)):
            # names.append('control_var' + str(i + 1))
            input_tuple.update({'control_var' + str(i + 1): [self.framework.control_variables.get_values_from_indices(u[i])]})
        data = pd.DataFrame(input_tuple)
        self.csv_input_file = pd.concat((self.csv_input_file, data), axis=0, ignore_index=True)
        out = pd.concat((self.time_csv, self.csv_input_file), axis=1)
        out.to_csv(self.csv_file_name, index=False)

    def set_inputs_from_list(self, u_list):
        # input_tuple = {}
        names = []
        u_list_t = np.zeros(u_list.shape, dtype=float)
        for i in range(u_list.shape[0]):
            u_list_t[i, :] = self.framework.control_variables.get_values_from_indices(u_list[i, :])
        for i in range(len(self.framework.control_variables)):
            # names.append('control_var' + str(i + 1))
            # input_tuple.update({'control_var' + str(i + 1): (u_list[:, i])})
            names.append('control_var' + str(i + 1))
        # data = pd.DataFrame(u_list_t, columns=names)
        # out = pd.concat((self.time_csv, data), axis=1, ignore_index=True)
        out = np.concatenate((self.time_data.reshape(-1, 1), u_list_t), axis=1)
        names.insert(0, 'time')
        # out.columns = names
        # out.to_csv(self.csv_file_name, index=False)
        with open(os.path.join(self.tmpdir, 'sim_input.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(names)
            # Write data
            for row in out:
                writer.writerow(row)
        return out

    def set_variable_filter(self, var_array: list | variable_array):
        if isinstance(var_array, variable_array):
            var_array = [variable_array]
        filter = 'variableFilter="'
        for v in var_array:
            if isinstance(v, variable_array):
                for var in v:
                    filter += var.name + '|'
            else:
                for var in v:
                    filter += var + '|'
        return filter[:-1] + '"'

    def get_sim_variables(self):
        xml_doc = xmlparser.parse(os.path.join(self.tmpdir, 'sim_init.xml'))
        scalar_variables = xml_doc.getElementsByTagName('ScalarVariable')
        sim_vars = []
        params = []
        input_vars = []
        state_vars = []
        aliases = {}
        for var in scalar_variables:
            if "Sta" in var.getAttribute('classType'):
                state_vars.append(var.getAttribute('name'))
            elif var.getAttribute('causality').lower() == 'parameter':
                params.append(var.getAttribute('name'))
            elif var.getAttribute('causality').lower() == 'input':
                input_vars.append(var.getAttribute('name'))
            elif var.getAttribute('alias').lower() == 'noalias':
                sim_vars.append(var.getAttribute('name'))
            else:
                aliases[var.getAttribute('name')] = var.getAttribute('aliasVariable')

        return {'sim_vars': sim_vars, 'state_vars': state_vars, 'input_vars': input_vars,
                'parameter_vars': params, 'aliases': aliases}

    def set_parameters(self, x: np.ndarray):
        # load XML file to modify parameter value
        xml_file = xmlET.parse(os.path.join(self.tmpdir, 'sim_init.xml'))
        variables = xml_file.findall('DefaultExperiment')

        parameter_names = self.variable_array.get_name()

        for (value, name) in zip(x, parameter_names):
            for var in variables:
                if var.get('name') == name:
                    var.find('Real').set('start', value)

        xml_file.write(os.path.join(self.tmpdir, 'sim_init.xml'))

    def get_parameters(self, params: list | np.ndarray | str = None):
        if params is None:
            return self.get_parameters(self.get_sim_variables()['parameter_vars'])

        if isinstance(params, list) | isinstance(params, np.ndarray):
            out = np.empty((len(params)), dtype=float)
            for i, param in enumerate(params):
                out[i] = self.get_parameters(param)
            return out

        xml_doc = xmlET.parse(os.path.join(self.tmpdir, 'sim_init.xml'))
        var = xml_doc.find(".//ScalarVariable[@name='" + params + "']")
        value = var.find('Real').get('start')
        return value