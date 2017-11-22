from Graph import*
from lea import*
import random as rd
import numpy as np
import Operations as op
import qubit_class as qubit
import Gates as g
import qutip as qt
from sympy import*


class Agent(Graph):
    def __init__(self, n_p, n_a, state, gamma, label_increment=1):
        super().__init__()
        self.constructgraph(n_p, n_a, state, step=label_increment)
        self.h = np.zeros((n_p, n_a), dtype=float)  # weight matrix initialized to all zeros
        self.gamma = gamma

    def excitepercept(self):
        """
        :return:  returns percept label vertex object that has been excited
        """

        # The labels in percept_list are placed randomly i.e 'p0' is not necessarily in the first place
        percept_list = self.perceptlabels()
        r = rd.randint(0, len(percept_list)-1)
        key_value = percept_list[r]
        self.vertices[key_value].graph_state = 1
        return key_value

    def constructgraph(self, n_p, n_a, state, step=1):
        """
        :param n_p: This is an integer for the number of percepts
        :param n_a: This is an integer for the number of actuators
        :param state: This is the state to be held initially at a vertex could be classical or quantum mechanical
        :param step: One might need the labels for the percepts and the actuators to go in increments other than
        one
        :return:
        """
        percept_labels = op.generateDictKeys('p', n_p, step)
        actuator_labels = op.generateDictKeys('a', n_a)

        for i in percept_labels:
            self.add_vertex(state, i)
        for j in actuator_labels:
            self.add_vertex(state, j)

        for m in percept_labels:
            for n in actuator_labels:
                self.add_edge(m, 0, n, 0, 1)

    def weightmatrix(self):
        """
        :return: Returns the weight matrix for the graph
        """
        percept_list = self.perceptlabels()
        actuator_list = self.actuatorlabels()
        for p in range(0, len(percept_list)):
            for a in range(0, len(actuator_list)):
                self.h[p, a] = self.vertices[percept_list[p]].get_weight(self.vertices[actuator_list[a]])
        return self.h

    def exciteactuator(self, label):
        """
        :param label: label for the actuator to be excited
        :return:
        """
        self.vertices[label].graph_state = 1

    def deexcitepercept(self, key_value, val):
        """
        :param key_value: The label of the vertex to de-excite
        :param val : if val is 0, then method assumes you have a classical agent if val is 1 then it de-excites
        a the whole graph
        :return:
        """
        try:
            if val == 0:
                self.vertices[key_value].graph_state = 0
            if val == 1:
                labels = self.getlabels()
                for lb in labels:
                    self.vertices[lb].graph_state = qubit.Qubit(0)
        except ValueError:
            print("Please enter 0 or 1 for second argument")

    def perceptlabels(self):
        percept_list = sorted(list(filter(lambda k: k[0] == 'p', self.vertices)))
        return percept_list

    def actuatorlabels(self):
        actuator_list = sorted(list(filter(lambda k: k[0] == 'a', self.vertices)))
        return actuator_list

    def getlabels(self):
        """
        Get all the labels for the vertices (percepts and actuators)
        :return: List of all the labels
        """
        l = []

        for v in self.vertices:
            l.append(v)
        return l

    def randomwalk(self, key_value, interaction_action=None):
        """
         This method carries out the random walk for the agent.
        :param key_value: key value for percept that has just been excited
        :param interaction_action: This is when agents are interacting, this is the perceived percepts
        :return:
        """
        tmp = 0
        p = rd.random()
        actuator_weights = {}
        # get a list of actuator labels
        actuator_list = self.actuatorlabels()

        # get a list of weights for a specific percept and normalize them
        for v in actuator_list:
            actuator_weights[v] = self.vertices[key_value].adjacent[self.vertices[v]]

        for i in actuator_weights:
            tmp += actuator_weights[i]

        for j in actuator_weights:
            actuator_weights[j] = (actuator_weights[j]/tmp)*100.00

        r = self.calculateblockefficiency()

        picked_label = self.update(actuator_weights, key_value, interaction_action)

        stringlist = {'picked label': picked_label, 'actuator labels': actuator_list,
                      'percept label': key_value, 'actuator weights': actuator_weights, 'random number': p,
                      'block efficiency': r}

        returnstring = 'Percept Label used: {percept label} \n Picked label : {picked label} ' \
            '\n Random num : {random number} \n Actuator Labels : {actuator labels} ' \
            '\n Actuator Weight: {actuator weights}'.format(**stringlist)

        data_summary = {'summary':stringlist, 'summary string': returnstring}

        return data_summary

    def calculateblockefficiency(self):
        """
        Calculates the block efficiency for the agent
        :return:
        """
        weightmatrix = self.weightmatrix()
        r = 0
        marginal_prob = 0
        rows, columns = weightmatrix.shape
        for i in range(0, rows):
            for j in range(0, columns):
                marginal_prob += weightmatrix[i, j]
            r += weightmatrix[i, i]/(marginal_prob*rows)
            marginal_prob = 0

        return r

    def forget(self):
        """
        Updates the weights of the graph by  reducing them using the gamma factor
        :return:
        """
        a_list = self.actuatorlabels()
        p_list = self.perceptlabels()
        for s in p_list:
            for v in a_list:
                f = self.vertices[s].adjacent[self.vertices[v]]
                self.vertices[s].adjacent[self.vertices[v]] = f-self.gamma*(f-1)

    def update(self, probdist, key_value='', interaction_action=None):
        """
        Picks the actutator according to some probability and does the update process
        :param probdist: This the list of probabilities
        :param key_value: This is the label of the picked percept
        :param interaction_action: This is the perceived percept when in the interacting case
        :return: Returns the picked label for the actuator after the update process
        """
        # Gets the number of the percept since each is labelled by a 'p' and a number e.g 'p2'
        percept_number = key_value[1:len(key_value)]

        # Pick a label
        label_flip = Lea.fromValFreqsDict(probdist)

        picked_label = label_flip.random()

        if interaction_action is None:
            # Agent did the right action
            if percept_number == picked_label[1:len(picked_label)]:
                w_right = self.vertices[key_value].adjacent[self.vertices[picked_label]]
                self.vertices[key_value].adjacent[self.vertices[picked_label]] = w_right + 1
                self.forget()
            # Agent did the wrong action
            else:
                self.forget()
        elif interaction_action != 'na': # This is the interacting case, you compare the seen action with your own proposed action
            action_number = key_value[1:len(interaction_action)]
            if action_number == picked_label[1:len(picked_label)]:
                w_right = self.vertices[key_value].adjacent[self.vertices[interaction_action]]
                self.vertices[key_value].adjacent[self.vertices[interaction_action]] = w_right + 1
                self.forget()
            else:
                self.forget()
        else:
            self.forget()

        return picked_label


class QuantumAgent(Agent):

    def __init__(self, n_p, n_a, state, gamma_1, labelincrement=1, decay_rate=0.2):
        super().__init__(n_p, n_a, state, gamma_1, label_increment=labelincrement)
        self.n = n_p + n_a
        self.decay_rate = decay_rate

    class Decorators(object):
        @classmethod
        def quantum_blockefficiency_decorator(cls, desired_key_val='p0', mode=True):
            """
            The purpose of this decorator is to restrict what percept you want to use when graphing
            block efficiency. The original function does not care.
            :param desired_key_val:
            :param mode:
            :return:
            """

            def wrapper_method(func):
                def restrict_to_specific_percept(self, list_op, key_val, version=2):
                    if mode:
                        if desired_key_val == key_val:
                            return func(self, list_op, desired_key_val, version)
                    else:
                        return func(self, list_op, key_val, version)

                return restrict_to_specific_percept

            return wrapper_method

        @classmethod
        def probdistribution_decorator(cls, desired_key_val='p0', mode=True):
            def wrapper_method(func):
                def restrict_to_specific_percept(self, rho_results, m_op, n_op, percept_label,
                                                 mesolve_results=None, p=None, qmodel=1):
                    if mode:
                        if desired_key_val == percept_label:
                            return func(self, rho_results, m_op, n_op, desired_key_val,
                                    mesolve_results, p, qmodel)
                    else:
                        return func(self, rho_results, m_op, n_op, percept_label,
                                    mesolve_results, p, qmodel)

                return restrict_to_specific_percept

            return wrapper_method

    def hint(self):
        """
        Constructs the interaction hamiltonian for the quantum walk
        :return ham: the interaction hamiltonian
        """
        # Get the weightmatrix which will now contain the couplings for the hamiltonian
        w = self.weightmatrix()

        perceptlabels = self.perceptlabels()
        actuatorlabels = self.actuatorlabels()
        ham = np.zeros((pow(2, self.n), pow(2, self.n)))
        operators = {'0': g.id(), '1': g.b3(), '2': g.b2()}

        # Construct the interaction term in the hamiltonian
        for p in range(0, len(perceptlabels)):
            for ac in range(0, len(actuatorlabels)):
                    hamiltonianstring = op.controlgatestring(self.n, ('1', p+1), ('2', len(perceptlabels)+ac+1))
                    temp = op.superkron(operators, val=1, string=hamiltonianstring)
                    ham += (temp + op.ctranspose(temp))*w[p, ac]
        return ham

    def hsite(self):
        """
        :return: Returns the on site hamiltonian
        """

        labels = self.getlabels()
        operators = {'0': g.id(), '1': np.dot(g.b3(), g.b2())}
        ham = np.zeros((pow(2, self.n), pow(2, self.n)))

        for l in range(0, len(labels)):
            hamstring = op.generatetensorstring(self.n, l+1)
            ham += op.superkron(operators, val=1, string=hamstring)
        return ham

    def getdensitymatrix(self):
        """
        gets the density matrix for the whole quantum graph
        :return: rho
        """
        labels = self.getlabels()
        rho = 1

        for i in labels:
            rho = np.kron(rho, self.vertices[i].graph_state.state)

        out = qt.Qobj(rho)
        return out

    def gethamiltonian(self):
        """
        :return: Returns Hamiltonian for system
        """

        h = self.hint() + self.hsite()
        out = qt.Qobj(h)
        return out

    def measurement_operators(self):
        """
        :return: Returns a dictionary of measurement operators used for the measurement step. There is an assumed
        labeling. A measurement operator is labelled by the classical actuator that one desires to be measured.
        E.g the measurement operator labelled by 'a2' will be used to measure the third classical acutator.
        """
        operators = {'0': g.id(), '1': np.dot(g.b3(), g.b2())}
        perceptnum = len(self.perceptlabels())
        actuatorlist = self.actuatorlabels()
        list_op = {}
        actlist = range(0, len(actuatorlist))

        for a_1, a_2 in zip(actlist, actuatorlist):
            opstring = op.generatetensorstring(self.n, perceptnum + a_1 + 1)
            list_op[a_2] = op.superkron(operators, val=1, string=opstring)

        return list_op

    def noaction_operators(self):
        operators = {'0': g.id(), '1': g.b1()}
        perceptnum = len(self.perceptlabels())
        actuatorlist = self.actuatorlabels()
        list_op = {}
        actlist = range(0, len(actuatorlist))

        for a_1, a_2 in zip(actlist, actuatorlist):
            opstring = op.generatetensorstring(self.n, perceptnum + a_1 + 1)
            list_op['n'+a_2] = op.superkron(operators, val=1, string=opstring)

        return list_op

    def quantumwalk(self, key_value, hamiltonian, time, time_slices, m_op, n_op, p=None, oper=g.id(), c_op=[]
                    , model=1, action_seen=None):
        """
        :param key_value: Label for the excited percept
        :param hamiltonian: This is the Hamiltonian for the graph
        :param time: The time for which the quantum walk will occur
        :param time_slices: Number of time slices for the evolution
        :param p: Probability at which to stop the evolution
        :param m_op: List of measurement operators. The labelling for them could be different depending on what
        Hamiltonian i.e model was used
        :param n_op: List of measurement operators the give probabilities for no action
        :param oper: This is the operator that acts on the quantum percept. It's main purpose is to produce a
        superposition of some kind. If one uses the Hadamard matrix then this will produce an equal superposition of
        two classical percepts
        :param c_op: This is a list of lindblad operators for the noisy quantum evolution
        :param model: If model is 1 then the first model graph is used otherwise we use density matrix for
        second quantum model
        :param action_seen: If we have two quantum agents interacting
        :return: Returns a dictionary. One entry has a summary of the results for the quantum walk like what percept
        was excited and what actuator was picked. The other entry has a quantum object that comes from the mesovle method
        used to evolve the density matrix
        """

        # This happens for the first quantum model
        if model == 1:
            q = qubit.Qubit(1)
            q.operator(oper)
            self.vertices[key_value].graph_state = q
        # For this second quantum model, every percept begins in superposition except for actuators
        # Then we get the picked percept and measure it and put the state back into the graph
        else:
            q = qubit.Qubit(0)
            q.operator(g.h())
            for pl in self.perceptlabels():
                self.vertices[pl].graph_state = q
            picked_percept = self.vertices[key_value].graph_state
            picked_percept.measure()
            # If the measurement gives 1 then we increment the percept label by one.
            if np.array_equal(picked_percept.state, g.b4()) is True:
                percept_number = str(int(key_value[1:len(key_value)]) + 1)
                new_key_value = 'p'+percept_number
                key_value = new_key_value

        rho = self.getdensitymatrix()
        t = np.linspace(0, time, time_slices)
        qrho = qt.Qobj(rho)

        # get projection operators for quantum evolution and make them quantum objects
        operators = list(op.makeQobj(m_op).values())
        # get list of quantum objects from list of lindblad operators
        c_op = op.makeQobj(c_op)
        opt = qt.Options(rhs_reuse=True, store_states=True)
        expectation_values = qt.mesolve(hamiltonian, qrho, t, c_op, operators, options=opt)

        #  Get probability distribution for the actuators
        prob_dis = self.probdistribution(expectation_values, m_op, n_op, key_value, expectation_values,
                                         qmodel=model)
        r = self.quantumblockefficiency(prob_dis, key_value, version=model)
        # Functions in class lea require number between 1 and 100 for probability
        for keys in prob_dis:
            prob_dis[keys] *= 100
        picked_label = self.update(prob_dis, key_value, interaction_action=action_seen)

        #  De-excite the whole graph
        self.deexcitepercept(key_value, 1)
        
        # Take the probabilities back to be within 0 and 1
        for keys in prob_dis:
            prob_dis[keys] /= 100

        stringlist = {'picked_percept': key_value,  'picked_actuator': picked_label, 'prob_dist': prob_dis,
                      'blocking_efficiency': r}

        summary = 'Picked percept: {picked_percept} \nPicked actuator: {picked_actuator} \n' \
            'Probability distribution: {prob_dist}  \nBlocking' \
                  ' Efficiency: {blocking_efficiency}\n' \
                  '------------------------------------------------------------------------------------'.format(**stringlist)

        data_summary = {'summary data': summary, 'expect_vals': expectation_values, 'block_efficiency': r,
                        'summary': stringlist}

        return data_summary

    @Decorators.probdistribution_decorator(desired_key_val='p0', mode=False)
    def probdistribution(self, rho_results, m_op, n_op, percept_label,  mesolve_results = None, p=None, qmodel=1):
        """
         Returns a list of probabilities determined by all the operators one could measure on the actuators
         The determines a discrete probability distribution
        :param rho: The density matrix to be used to get the discrete probabililty distribution for the measurement
        procedure
        :param percept_label: This is the label for the picked percept
        :param n_op : Operators that correspond to no action being taken
        :param m_op: This is a list of measurement operators for used in the measurement step after quantum walk
        :return: list of probabilities for measuring each actuators
        :param : model: We have different no action operators for the two different models so we
         need to take that into consideration. model=1 is the first quantum model and model=2 is the
         second quantum model
        """
        # This gets the labels for the actuators used
        oper_labels = m_op.keys()
        noaction_labels = n_op.keys()
        probdis = {}
        percept_number = int(percept_label[1:len(percept_label)])
        rho = 0
        if qmodel == 1:
            time_index, measure_time = self.get_time(mesolve_results.expect[percept_number], mesolve_results.times, p)
            rho = rho_results.states[time_index].full()
        else:
            if percept_number == 0:
                time_index, measure_time = self.get_time(mesolve_results.expect[0], mesolve_results.times,
                                                      p)
                rho = rho_results.states[time_index].full()
            else:
                time_index, measure_time = self.get_time(mesolve_results.expect[3], mesolve_results.times,
                                                         p)
                rho = rho_results.states[time_index].full()

        if mesolve_results is None:
            temp = 0
            probdis = {a_2: np.trace(np.dot(m_op[a_2], rho)).real for a_2 in oper_labels}
            for n in noaction_labels:
                temp += np.trace(np.dot(n_op[n], rho)).real
            probdis['na'] = temp
        else:
            temp = 0
            num_operators = range(0, mesolve_results.num_expect)
            for i, a in zip(num_operators, oper_labels):
                    exp_val = mesolve_results.expect[i][time_index]
                    if exp_val < 0:
                        probdis[a] = 0
                    else:
                        probdis[a] = mesolve_results.expect[i][time_index]
            if qmodel == 1:
                for n in noaction_labels:
                    temp += np.trace(np.dot(n_op[n], rho)).real
                probdis['na'] = temp - 1
            else:
                for n in noaction_labels:
                    temp += np.trace(np.dot(n_op[n], rho)).real
                probdis['na'] = temp
       # print('distribution:', probdis, 'picked percept:', percept_label, 'temp:', temp)
        # actuator excited
        return probdis

    def getlindbladoperators(self):
        """
        :param decay:  Decay parameter for the qubits
        :return: Returns a list of lindblad operators
        """
        perceptlabels = self.perceptlabels()
        actuatorlabels = self.actuatorlabels()
        operators = {'0': g.id(), '1': g.b3(), '2': g.b2()}
        list_op = []
        for p in range(0, len(perceptlabels)):
            for ac in range(0, len(actuatorlabels)):

                    hamiltonianstring = op.controlgatestring(self.n, ('1', p+1), ('2', len(perceptlabels)+ac+1))
                    temp = op.superkron(operators, val=1, string=hamiltonianstring)
                    list_op.append(np.sqrt(self.decay_rate)*temp)

        return list_op

    def get_time(self, prb, times, p):
        """
        :param prb:  array of probabilities or expectation values
        :param times: list of time slices
        :param p The benchmark, could be object None in which case we pick the maximum value
        :return: time at which elements in prb get above p or last time if elements never
        reach above and the index of that time in the list
        """
        t = 0
        ind = 0
        length = len(times)
        times = times.tolist()
        probabilities = prb.tolist()

        if p is None:
            maximum = max(probabilities)
            ind = probabilities.index(maximum)
            t = times[ind]
            return ind, t
        else:
            for i in range(len(prb)):
                if prb[i] >= p:
                    t = times[i]
                    ind = times.index(t)
                    break
                else:
                    t = times[length-1]
                    ind = times.index(t)
            return ind, t

    def correlation_operators(self):
        """
        This produces correlation operators used to calculate blocking efficiency for
        :return correlation_oper: This is a list of correlation operators used for calculating
        blocking efficiency
        """
        measure_opers = self.measurement_operators()
        correlation_oper = {}

        for m in measure_opers:
            label_length = len(m)
            index1 = m.index('a')
            if m[1:index1] == m[index1+1:label_length]:
                correlation_oper[m] = measure_opers[m]

        return correlation_oper

    def measure_oper(self, oper_string, val=0):
        """
        :return : Returns a specific measurement operators. The label of an operator is got by concatenating the
        string label for the classical percept encoded and the label of the classical actuator it connects to by an
        edge. An accepted value of a string is 'p2a1'. This refers to the third classical percept being excited
        leading to the second classical actuator being excited
        """
        if val == 0:
            if isinstance(oper_string, str):
                temp = self.measurement_operators()
                out = temp[oper_string]
            else:
                print("Value of input parameter must be string of the kind. Also check the label of operators used")
        elif val == 1:
            if isinstance(oper_string, str):
                temp = self.noaction_operators()
                out = temp[oper_string]
        return out

    @Decorators.quantum_blockefficiency_decorator(desired_key_val='p0', mode=False)
    def quantumblockefficiency(self, list_op, key_val, version=2):
        """
        :param: list_op: This will be a dictionary of probabilities
        This will calculate the quantum analogue of the blocking efficiency
        :param: key_val: Picked percept label
        :param: version: This determines whether we are dealing with the first quantum model i.e version=1
        or the second quantum version i.e version=2
        :return: Returns the blocking efficiency
        """
        try:
            r = 0
            percept_number = key_val[1:len(key_val)]
            if isinstance(list_op, dict):
                if version == 1:
                    for label in list_op:
                        actuator_number = label[1:len(label)]
                        if percept_number == actuator_number:
                            r = list_op[label]
                else:
                    for label in list_op:
                        if label != 'na':
                            a_index = label.index("a")
                            actuator_number = label[a_index + 1:len(label)]
                            if percept_number == actuator_number and key_val == label[:2]:
                                r = list_op[label]
            return r
        except TypeError:
            print("First parameter must be a dictionary of operators")


class QuantumAgent_1(QuantumAgent):
    """
    This class was constructed because it has a different hamiltonian for the quantum walk. Here two percepts are encoded
    in a superposition of a quantum  percept. This model ultimately can't be totally encapsulated by a classical model
    It simulates a classical graph that is initialized in the _init_ method. This class has the possibiility of
    producing quantum reinforcement learning.
    """
    def __init__(self, n_p, n_a, state, gamma_2, percept_increment=2, decay_rate=0.2):
        """
        :param n_p: The number of quantum percepts. This means we can simulate 2*n_p classical percpets
        :param n_a: The number of quantum actuators. We also measure them and do not talk about quantum mechanical
        superposition of classical actuators
        :param state: This is the initial density matrix for put on the quantum graph
        :param gamma_2: This the decay parameter for noisy quantum evolution
        """
        super().__init__(n_p, n_a, state, gamma_2, labelincrement=percept_increment,
                         decay_rate=decay_rate)
        self.w = np.ones((n_p*2, n_a))
        self.n = n_p + n_a
        self.gamma_2 = gamma_2

    def hint(self):
        """
        :return: Interaction Hamiltonian for the model that has potentially purely quantum mechanical reinforcement
        learning. This interaction Hamiltonian cares about the quantum state of the percept which could be in a s
        superposition of two classical percepts.
        """
        perceptlabels = self.perceptlabels()
        actuatorlabels = self.actuatorlabels()
        ham = np.zeros((pow(2, self.n), pow(2, self.n)))
        operators = {'0': g.id(), '1': g.b1(), '2': g.b4(), '3': g.b3()}

        # Construct the interaction term in the Hamiltonian
        for p in range(0, len(perceptlabels)):
            for ac in range(0, len(actuatorlabels)):
                # String for the percept in the zero state
                hamiltonianstring_zero = op.controlgatestring(self.n, ('1', p+1), ('3', len(perceptlabels)+ac+1))
                # String for the percept in the one state
                hamiltonianstring_one = op.controlgatestring(self.n, ('2', p+1), ('3', len(perceptlabels)+ac+1))

                temp_zero = op.superkron(operators, val=1, string=hamiltonianstring_zero)
                ham += (temp_zero + op.ctranspose(temp_zero))*self.w[2*p, ac]

                temp_one = op.superkron(operators, val=1, string=hamiltonianstring_one)
                ham += (temp_one + op.ctranspose(temp_one))*self.w[2*p+1, ac]

        return ham

    def measurement_operators(self):
        """
        The Measurment operators encoded both what percept was excited and what actuator was excited. In the quantum
        model one does not measure the percept instead one lets the superposition evolve. As a consequence the operators
        one is interested in must be labelled by the percepts and the actuators. E.g the label 'p0a0' means the first
        percept was excited and the first actuator was excited. This can also be reformulated to mean we label the
        measurement operators by the edge connecting the actuator and the percept envolved. So there should be as many
        edges as there are measurement operators
        :return: Returns a list measurement operators list_op
        """
        list_op = {}
        perceptlabels = self.perceptlabels()
        actuatorlabels = self.actuatorlabels()
        operators = {'0': g.id(), '1': g.b1(), '2': g.b4(), '3': g.b1()}
        perceptlist = range(0, len(perceptlabels))
        actlist = range(0, len(actuatorlabels))
        np = len(perceptlabels)
        na = len(actuatorlabels)

        for p, p1 in zip(perceptlist, perceptlabels):
            for ac, ac1 in zip(actlist, actuatorlabels):
                percept_counter = 2 * p

                # String for the percept in the zero state
                hamiltonianstring_zero = op.generatehamiltoniantring(np, '1', onestring=True, pos=p,
                                                                     pad='0')
                # Adding string for the actuators
                hamiltonianstring_zero += op.generatehamiltoniantring(na, '2', onestring=True, pos=ac,
                                                                      pad='3')
                # String for the percept in the one state
                hamiltonianstring_one = op.generatehamiltoniantring(np, '2', onestring=True, pos=p,
                                                                    pad='0')
                # Adding string for the actuators
                hamiltonianstring_one += op.generatehamiltoniantring(na, '2', onestring=True, pos=ac,
                                                                     pad='3')

                list_op['p'+str(percept_counter)+ac1] = op.superkron(operators, val=1, string=hamiltonianstring_zero)

                percept_counter += 1

                list_op['p'+str(percept_counter)+ac1] = op.superkron(operators, val=1, string=hamiltonianstring_one)

        return list_op

    def noaction_operators(self):
        """
        This produces the operator that is responsible for calculating the probability of no action. Since an
        excitation on the actuator qubit and no excitation on any other actuator qubit corresponds to a
        specific action taken, no excitation on any actuator qubit corresponds to no action taken. Contrast
        this with the first model. There an excitation occurs on an actuator qubit but we do not care what is
        happening to the other qubits. So when we construct no action operators we have to construct operators
        for each actuator qubit
        :return: A dictionary of operators
        """
        operators = {'0': g.id(), '1': g.b1(), '2': g.b4()}
        perceptnum = len(self.perceptlabels())
        actuatorlist = self.actuatorlabels()
        actnum = len(actuatorlist)
        opstring = '0'*perceptnum
        opstring += '1'*actnum
        opstring1 = '0'*perceptnum
        opstring1 += '2'*actnum
        noaction_op={}
        noaction_op['na0'] = op.superkron(operators, val=1, string=opstring)
        noaction_op['na1'] = op.superkron(operators, val=1, string=opstring1)


        return noaction_op

    def update(self, probdist, key_value='', interaction_action=None):
        """
                Picks the actutator according to some probability and does the update process
                :param probdist: This the list of probabilities
                :param key_value: This is the label of the picked percept
                :return: Returns the picked label for the actuator after the update process
        """

        # Pick a label
        label_flip = Lea.fromValFreqsDict(probdist)

        picked_label = label_flip.random()
        # print('picked label in update:', picked_label)

        if picked_label != 'na':
            index1 = picked_label.index("a")
            percept_number = int(picked_label[1:index1])  # Finds the number of the percept
            actuator_number = int(picked_label[index1+1:len(picked_label)])
            # print('percept number in update:', percept_number)
            # print('actuator number in update: ', actuator_number)
            if percept_number == actuator_number:
                self.w[percept_number, actuator_number] += 1
                # print(self.w)
                self.forget()
            else:
                self.forget()
        else:
            pass
            # self.forget
        return picked_label

    def forget(self):
        """
        Does the forgetting. Adjusts the weight matrix
        """
        size = self.w.shape

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                f = self.w[i, j]
                self.w[i, j] = f - self.gamma_2*(f-1)




