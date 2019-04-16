# -*- coding: utf-8 -*-

__author__    = "Christian Richter"
__copyright__ = "Copyright 2019, TU Dresden"
__license__   = "GPL"
__credits__   = ["Christian Richter"]
__email__     = "christian.richter1@tu-dresden.de"
__project__   = "sysrl"
__version__   = "0.1.0"


import os
import gym
import logging
import pyfmi
import warnings
import datetime
import numpy as np

from gym import spaces
from collections import OrderedDict
from pyfmi import load_fmu


__all__ = ['FmiEnvironment']


class FmiEnvironment(gym.Env):
    """
    Superclass for all FMU environments.
    Simulate a FMU CS2.
    All inputs variables are actions.
    All output variables are observations.
    Link to pyfmi library: http://www.jmodelica.org/assimulo_home/pyfmi_1.0/##
    """

    random_state = np.random.RandomState()

    def __init__(self, fmu, observation_limits = {}):
        if not os.path.exists(fmu):
            raise IOError("File %s does not exist" % fmu)

        self.model_name = os.path.basename(fmu).split('.')[0]
        self.model = load_fmu(fmu, kind='CS', log_file_name=self._get_log_file_name())

        self.logger = logging.getLogger(self.model_name)
        self.logger.setLevel(logging.DEBUG)

        self.config = {
            'actions': None,
            'observations': None,
            'observation_limits' : observation_limits
        }
        
        # create action space
        self._create_action_space()
        
        # create observation space
        self._create_observation_space()        
        
        # start time, step size
        if isinstance(self.model, pyfmi.fmi.FMUModelBase2):
            self.tau = self.model.get_default_experiment_step()
            self.start = self.model.get_default_experiment_start_time()
            self.stop = self.model.get_default_experiment_stop_time()
        else:
            self.tau = 0.02
            self.start = 0
            self.stop = 1

        self.done = True

    
    def seed(self, seed=None):
        if seed is None:
            self.random_state = np.random.RandomState()
        elif isinstance(seed, int):
            self.random_state = np.random.RandomState(seed)
        else:
            raise TypeError("Seed got wrong type '%s'." %type(seed))


    def _create_action_space(self):
        # get input variables
        if isinstance(self.model, pyfmi.fmi.FMUModelBase2):
            inputs = self.model.get_input_list() 
        else:
            inputs = self.model.get_model_variables(causality=0)

        if len(inputs) != 1:
            print("Inputs: ", inputs)
            raise Exception("At the moment only FMU's with exact 1 input variable are allowed!")

        names = list(inputs.keys())

        # check type of scalar variable and create action space
        if inputs[names[0]].type == 0:                                      # Real
            warnings.warn("Continuous action space not tested yet!")
            min_value = self.model.get_variable_min(names[0])
            max_value = self.model.get_variable_max(names[0])
            self.action_space = spaces.Box(min_value, max_value, shape=(1,), dtype=np.float32)
        elif inputs[names[0]].type == 1 or inputs[names[0]].type == 2:      # Integer or Boolean
            min_value = self.model.get_variable_min(names[0])
            max_value = self.model.get_variable_max(names[0])
            space = max_value - min_value + 1
            warnings.warn("Discrete action space not tested yet!")
            self.action_space = spaces.Discrete(space)
        else:
            raise Exception("Input '%s' got wrong type. Must be 'Real',"\
                            "'Integer' or 'Boolean'." % names[0])
        self.config['actions'] = names

        
        
    def _create_observation_space(self):
        # get output variables
        if isinstance(self.model, pyfmi.fmi.FMUModelBase2):
            outputs = self.model.get_output_list() 
        else:
            outputs = self.model.get_model_variables(causality=1)
        
        observations = []
        upper_limits = []
        lower_limits = []

        for name, sv in outputs.items():
            if sv.type != 0: raise Exception("All outputs variables must be from type 'Real'. Got Type: %s." %sv.type)
            if name in self.config['observation_limits']:
                lower_limits.append(self.config['observation_limits'][name][0])
                upper_limits.append(self.config['observation_limits'][name][1])
            else:
                lower_limits.append(-np.inf)
                upper_limits.append(np.inf)
            observations.append(name)
    
        self.observation_space = spaces.Box(np.array(lower_limits), np.array(upper_limits), dtype=np.float32)
        self.config['observations'] = observations


    def _get_log_file_name(self):
        log_date = datetime.datetime.utcnow()
        log_file_name = "{0}-{1}-{2}_{3}.txt".format(log_date.year, log_date.month, log_date.day, self.model_name)
        return log_file_name
    

    def _set_init_parameter(self):
        parameters = self.initial_parameters()
        self.model.set(list(parameters), list(parameters.values()))


    def initial_parameters(self):
        """ Can be reimplemented """ 
        return {}


    def step(self, action):
        if self.done == True:
            self.reset()
            #self._restart_simulation()
        else:
            self.start = self.stop
            self.stop += self.tau
        
        # TODO: Aktuell nur eine Action möglich!
        self.model.set(self.config['actions'][0], action)
        self.model.do_step(self.start, self.tau, True)

        # get the current state
        self.state = self._get_state()

        self.done = self.is_done()

        reward = self.calc_reward()
        
        state = np.array(list(self.state.values()))
        
        return state, reward, self.done, {}
    
    
    def calc_reward(self):
        """
        This function must be reimplemented by an inheriting environment.
        
        :return: Integer or Float
        """
        raise NotImplementedError
    
    def is_done(self):
        """
        This function must be reimplemented by an inheriting environment.
        
        :return: bool
        """
        raise NotImplementedError
    

    def _restart_simulation(self):
        """
        Resets the Modellica model to its original state
        """
        #logger.debug("restart simulation")
        self.model.reset()
        #self.done = False 
        self._set_init_parameter()
        self.start = 0
        self.stop = self.tau
        if isinstance(self.model, pyfmi.fmi.FMUModelBase2):
            self.model.setup_experiment()
        self.model.initialize()


    def _get_state(self):
        """
        Get the final value of states
        :return: states
        """
        return OrderedDict([(k, self.model.get(k)[0]) for k in self.config['observations']])
    
    
    '''
    def reset1(self):
        self._restart_simulation()
        self.state = self._get_state()
        return np.array(list(self.state.values()))
    '''

    def reset(self):
        self.model.reset()
        #self.done = False 
        self._set_init_parameter()
        self.start = 0
        self.stop = self.tau
        if isinstance(self.model, pyfmi.fmi.FMUModelBase2):
            self.model.setup_experiment()
        self.model.initialize()
        self.state = self._get_state()
        return np.array(list(self.state.values()))












if __name__ == '__main__':
    """ For testing and development only! """


    #FMU_PATH = "D:/02_Projekte/fmi4gym/inverted_pendulum/Pendel_Komponenten_Pendulum.fmu"
    FMU_PATH = "D:/02_Projekte/fmi4gym/inverted_pendulum/Pendel_Komponenten_Pendulum2.fmu"
    
    model = load_fmu(FMU_PATH, kind='CS')
    #model.instantiate_slave()

    #model.setup_experiment()
    model.set('m_load', 10)
    model.set('phi1_start', 1.6)
    model.initialize()
   
    
    #print(model.get('phi1'))
    print(model.get('m_load'))
    print(model.get('phi1'))
    
    print(model.get_variable_max('phi1', 1))

    #env = FmuEnvironment(FMU_PATH)
    
    #print(x_threshold)
    #print(theta_threshold_radians)

    #print(env.action_space)
    #print(env.observation_space)
    
    #print(env._get_log_file_name())

    # Notiz: 
