from concurrent.futures import ThreadPoolExecutor
from cpp import DynamicModel
from matplotlib import pyplot as plt
from import_mesh import read_mesh_to_edges
from import_mesh import get_nodes_for_line
from import_mesh import read_mesh_pos_as_dofs

from asseble_stiffness_matrix import assemble_global_stiffness
import numpy as np
import datetime
import os, threading
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation



edges,number_of_nodes = read_mesh_to_edges("test_5.msh")
BC_nodes = get_nodes_for_line("test_5.msh", "BC")
end_nodes = get_nodes_for_line("test_5.msh", "end")
nodes = read_mesh_pos_as_dofs("test_5.msh")

print(f"number of nodes: {number_of_nodes} ")
print(f"excitation nodes: {BC_nodes}")
print(f"end nodes: {end_nodes}")

print('start assembling stiffness matrix')
K = assemble_global_stiffness(edges, number_of_nodes)
print('finished assembling stiffness matrix')

rows, cols = K.nonzero()
values = K.data
nrows, ncols = K.shape
triplets = np.vstack((rows, cols, values)).T.astype(np.float64) 


animation = False

def run_single_simulation(args):
    steps,dt, strart_f, end_f, damping_div = args
    print(f"Running simulation with start_f: {strart_f}, end_f: {end_f}, damping_div: {damping_div}")
    model_1 = DynamicModel.DynamicModel(triplets,nrows,ncols,BC_nodes,end_nodes,number_of_nodes, damping_div * 2000000)
    print(model_1.mass_per_dof)
    exc_log_1, end_log_1, T_log_1 = model_1.run_simulation(steps, dt, strart_f, end_f, 10)
    
    if animation:
        u_log = model_1.u_log
        return exc_log_1, end_log_1, T_log_1, u_log
    else:
        return exc_log_1, end_log_1, T_log_1


args = [
        (10**6, 10**-7, 90, 90,  350),
        (10**6, 10**-7, 90, 90,  700),
        (10**6, 10**-7, 90, 90,  1050),
        (10**6, 10**-7, 90, 90,  1400),
        (10**6, 10**-7, 90, 90,  1750),
        (10**6, 10**-7, 90, 90,  2100)
        ]

with ThreadPoolExecutor(max_workers = 3) as executor:
    results = list(executor.map(run_single_simulation, args))




# for animation 
if animation:
    list_a, list_b, list_c, u_log = zip(*results)
    pickle.dump(u_log, open('u_log.pkl', 'wb'))
else:
    pickle.dump(results, open('end.pkl', 'wb'))


