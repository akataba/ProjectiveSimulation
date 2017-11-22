import ClipSpace as cp
import Data as da
import qubit_class as qubit
import Gates as g
import numpy as np
import Operations as op
from joblib import Parallel, delayed
from sympy import*
import abc
from qutip import *
from sympy.physics.quantum import *


#################################################################
# Classical Projective Simulator
#################################################################

class Simulator(object):
    """
    Abstract class for classical and quantum simulators
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def create_agents(self):
        pass

    @abc.abstractmethod
    def perform_walk(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def graph_work(self, x_label, y_label):
        pass

    @abc.abstractmethod
    def store_data(self, filename):
        pass

    @abc.abstractclassmethod
    def construct_filename(self):
        pass


class ClassicalSimulator(Simulator):
    """
    Perform Simulations with the classical projective agent
    """
    def __init__(self, no_agents, walks, n_percepts, n_actuators, interact=False, forget_factor=0.2):
        self.a_results = da.Data()
        if interact:
            self.b_results = da.Data()
        self.no_agents = no_agents
        self.walks = walks
        self.n_percepts = n_percepts
        self.n_actuators = n_actuators
        self.interact = interact
        self.forget_factor = forget_factor
        self.a = cp.Agent(self.n_percepts, self.n_actuators, 0, self.forget_factor)
        if self.interact:
            self.b = cp.Agent(self.n_percepts, self.n_actuators, 0, self.forget_factor)


    def perform_walk(self):

        for h in range(0, self.no_agents):
            for j in range(0, self.walks):
                percept_excited = self.a.excitepercept()
                a_summary = self.a.randomwalk(percept_excited)
                seen_action = a_summary['summary']['picked label']
                if self.interact:
                    b_summary = self.b.randomwalk(percept_excited, interaction_action=seen_action)
                self.a_results.add('r'+str(h), a_summary['summary']['block efficiency'])
                self.a_results.add_label('r'+str(h))
                if self.interact:
                    self.b_results.add('b'+str(h), b_summary['summary']['block efficiency'])
                    self.b_results.add_label('b'+str(h))

    # Graphing blocking efficiency

    def graph_work(self, x_label, y_label):
        self.a_results.xlabel = x_label
        self.a_results.ylabel = y_label
        self.a_results.get_colors()
        self.a_results.dataset_mean(str(self.no_agents)+' agents')
        self.a_results.dataset_stddev(str(self.no_agents)+' agents')
        self.a_results.graph_errorbars(x_axis=list(range(0, self.walks)), key=str(self.no_agents)+' agents',
                              key_1=str(self.no_agents)+' agents', label_1='First Agent', use_dict=True)
        if self.interact:
            self.b_results.xlabel = x_label
            self.b_results.ylabel = y_label
            self.b_results.get_colors()
            self.b_results.dataset_mean(str(self.no_agents) + ' agents')
            self.b_results.dataset_stddev(str(self.no_agents) + ' agents')
            self.b_results.graph_errorbars(x_axis=list(range(0, self.walks)), key=str(self.no_agents) + ' agents',
                              key_1=str(self.no_agents) + ' agents', label_1='Second Agent', use_dict=True)

    def store_data(self, filename):
        op.write_to_file(filename + '_first_agent.txt', self.a_results.dataset_average[str(self.no_agents) + ' agents'],
                     self.a_results.standard_deviation[str(self.no_agents) + ' agents'])
        if self.interact:
            op.write_to_file(filename + '_second_agent.txt', self.b_results.dataset_average[str(self.no_agents) + ' agents'],
                     self.b_results.standard_deviation[str(self.no_agents) + ' agents'])

    def construct_filename(self, file_stem, noise=False):
        """
        :param file_stem: If you need to specify absolute file path
        :param noise: If the simulations have noise in them then there will be a decay rate
        :return: Return file name which is either absolute filepath or
        just filename
        """
        filename = ''
        noagents = str(self.no_agents) + '_Agents_'
        nowalks = '_' + str(self.walks) + 'quantumwalks_'
        forgetfactor = '_forgetfactor_' + str(self.forget_factor).replace('.', '_') + '_'
        filename = noagents + nowalks + forgetfactor
        return file_stem + filename


class QuantumSimulatorOne(Simulator):
    """
    Perform Simulations with the first quantum model
    """
    def __init__(self, no_agents, quantum_walks, n_percepts, n_actuators, interact=False, forget_factor=0.2
                 , dissipative_factor=0.01, time_slices=1000, total_time=5, decay=0.01, noise=False):
        self.quantum_data = da.Data()
        if interact:
            self.quantum_data2 = da.Data()
        self.no_agents = no_agents
        self.quantum_walks = quantum_walks
        self.n_percepts = n_percepts
        self.n_actuators = n_actuators
        self.interact = interact
        self.forget_factor = forget_factor
        self.dissipative_factor = dissipative_factor
        self.time_slices = time_slices
        self.total_time = total_time
        self.decay = decay
        self.noise = noise
        self.quantum = 0
        self.quantum_2 = 0

    def perform_walk(self, h, data_object, data_object2=None, model_p=1):

        m_op = self.quantum.measurement_operators()
        n_op = self.quantum.noaction_operators()
        if self.noise:
             noise_op = self.quantum.getlindbladoperators()
        data_object_dict = {}
        for j in range(0, self.quantum_walks):
            percept_excited = self.quantum.excitepercept()
            ham = self.quantum.gethamiltonian()
            if self.noise:
                results = self.quantum.quantumwalk(percept_excited, ham, self.total_time,
                                              self.time_slices, m_op, n_op, c_op=noise_op
                                                   , model=model_p)
            else:
                results = self.quantum.quantumwalk(percept_excited, ham, self.total_time,
                                              self.time_slices, m_op, n_op, model=model_p)
            if self.interact:
                seen_action = results['summary']['picked_actuator']
                ham2 = self.quantum_2.gethamiltonian()
                results2 = self.quantum_2.quantumwalk(percept_excited, ham2, self.total_time,self.time_slices, m_op, n_op, action_seen=seen_action)
            if results['block_efficiency'] is None:
                continue
            if self.interact:
                if results2['block_efficiency'] is None:
                    continue
            data_object.add('block' + str(h), results['block_efficiency'])
            data_object.add_label('block' + str(h))
            if self.interact:
                data_object2.add('block' + str(h), results2['block_efficiency'])
                data_object2.add_label('block' + str(h))

        data_object_dict['data_object'] = data_object
        if self.interact:
            data_object_dict['data_object2'] = data_object2

        return data_object_dict

    def create_agents(self):
        self.quantum = cp.QuantumAgent(self.n_percepts, self.n_actuators, qubit.Qubit(0), self.forget_factor,
                                       decay_rate=self.decay)
        if self.interact:
            self.quantum_2 = cp.QuantumAgent(self.n_percepts, self.n_actuators, qubit.Qubit(0), self.forget_factor,
                                             decay_rate=self.decay)

    def graph_work(self, x_label, y_label, model):
        # The line below returns a list of data objects. One for each value of h
        if self.interact:
            quantum_results=Parallel(n_jobs=3, verbose=11)(delayed(self.perform_walk)(h, self.quantum_data,
                                                                             data_object2=self.quantum_data2,
                                                                        model_p=model) for h in range(0, self.no_agents))
        else:
            quantum_results = Parallel(n_jobs=3, verbose=11)(delayed(self.perform_walk)(h, self.quantum_data, model_p=model)for h in range(0, self.no_agents))

    # Put all the data objects into 1 data object

        for i in range(0, len(quantum_results)):
            for l in quantum_results[i]['data_object'].label:
                self.quantum_data.d[l] = quantum_results[i]['data_object'].d[l]

        if self.interact:
            for i in range(0, len(quantum_results)):
                for l in quantum_results[i]['data_object2'].label:
                    self.quantum_data2.d[l] = quantum_results[i]['data_object2'].d[l]

        self.quantum_data.xlabel = x_label
        self.quantum_data.ylabel = y_label
        self.quantum_data.get_colors()
        self.quantum_data.dataset_mean(str(self.no_agents)+' Agents')
        self.quantum_data.dataset_stddev(str(self.no_agents)+' Agents')

        if self.interact:
            self.quantum_data2.xlabel = x_label
            self.quantum_data2.ylabel = y_label
            self.quantum_data2.get_colors()
            self.quantum_data2.dataset_mean(str(self.no_agents)+' Agents')
            self.quantum_data2.dataset_stddev(str(self.no_agents)+' Agents')

        self.quantum_data.graph_errorbars(x_axis=list(range(self.quantum_walks)),key= str(self.no_agents)+' Agents',
                                 key_1=str(self.no_agents)+' Agents', use_dict=True, label_1='First Agent')
        if self.interact:
            self.quantum_data2.graph_errorbars(x_axis=list(range(self.quantum_walks)), key=str(self.no_agents) + ' Agents',
                                  key_1=str(self.no_agents) + ' Agents', use_dict=True, label_1='Second Agent')

    def store_data(self, filename):
        op.write_to_file(filename + 'first_agent.txt',
                         self.quantum_data.dataset_average[str(self.no_agents) + ' Agents'],
                         self.quantum_data.standard_deviation[str(self.no_agents) + ' Agents'])
        if self.interact:
            op.write_to_file(filename + 'second_agent.txt',
                             self.quantum_data2.dataset_average[str(self.no_agents) + ' Agents'],
                             self.quantum_data2.standard_deviation[str(self.no_agents) + ' Agents'])

    def construct_filename(self, file_stem='', noise=False):
        """
        :param file_stem: If you need to specify absolute file path
        :param noise: If the simulations have noise in them then there will be a decay rate
        :return: Return file name which is either absolute filepath or
        just filename
        """
        noagents = str(self.no_agents) + '_Agents_'
        nowalks = '_'+ str(self.quantum_walks) +'walks_'
        forgetfactor = '_forgetfactor_'+str(self.forget_factor).replace('.', '_') + '_'
        decay = '_decay_'+ str(self.decay).replace('.', '_')+ '_'
        if noise:
            filename = noagents +nowalks + forgetfactor + decay
        else:
            filename = noagents + nowalks + forgetfactor
        return file_stem + filename


class QuantumSimulatorTwo(QuantumSimulatorOne):
    """
     Perform simulations with the second quantum model
    """
    def __init__(self, no_agents, quantum_walks, n_percepts, n_actuators, interact=False, forget_factor=0.2
                 , dissipative_factor=0.01, time_slices=1000, total_time=5, decay=0.01, noise=False):
        super().__init__(no_agents, quantum_walks, n_percepts, n_actuators, interact, forget_factor
                 ,dissipative_factor, time_slices, total_time, decay, noise)

    def create_agents(self):
        # self.quantum = cp.QuantumAgent_1(self.n_percepts, self.n_actuators, qubit.Qubit(0), self.forget_factor)
        self.quantum = cp.QuantumAgent_1(self.n_percepts, self.n_actuators, qubit.Qubit(0), self.forget_factor)
        if self.interact:
            self.quantum_2 = cp.QuantumAgent_1(self.n_percepts, self.n_actuators, qubit.Qubit(0), self.forget_factor)



