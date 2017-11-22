from Datacollection import ClassicalSimulator, QuantumSimulatorOne, QuantumSimulatorTwo

if __name__ == "__main__":
    # c = ClassicalSimulator(200, 40, 2, 2, interact=False, forget_factor=0.2)
    # c.perform_walk()
    # c.graph_work('$n^{th} $ classical walk', 'blocking efficiency')
    # f= c.construct_filename('/home/amara/Python Files/Projective-Simulation/Projective'
    #                      ' Simulation/Data/Trial/', noise=False)
    # c.store_data(f)

    #######################################################################
    # Quantum Simulations (First Model)
    ########################################################################

    q = QuantumSimulatorOne(30, 10, 2, 2, interact=False, forget_factor=0.01, time_slices=1000,
                            total_time=5)
    q.create_agents()
    q.graph_work('$n^{th}$ quantum walk', 'blocking efficiency', model=1)
    f = q.construct_filename('/home/amara/Python Files/Projective-Simulation/Projective'
                         ' Simulation/Data/Trial/', noise=q.noise)
    q.store_data(f)

    #######################################################################
    # Quantum Simulations (Second Model)
    ########################################################################

    # q_1 = QuantumSimulatorTwo(30, 30, 1, 2, interact=False, forget_factor =0.01,
    #                           time_slices=1000, total_time=5)
    # q_1.create_agents()
    # q_1.graph_work('$n^{th}$ quantum walk', 'blocking efficiency', 2)

