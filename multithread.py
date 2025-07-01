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


animation = True

def run_single_simulation(args):
    steps,dt, strart_f, end_f, damping_div = args
    print(f"Running simulation with start_f: {strart_f}, end_f: {end_f}, damping_div: {damping_div}")
    model_1 = DynamicModel.DynamicModel(triplets,nrows,ncols,BC_nodes,end_nodes,number_of_nodes, damping_div)
    print(model_1.mass_per_dof)
    exc_log_1, end_log_1, T_log_1 = model_1.run_simulation(steps, dt, strart_f, end_f, 100)
    
    if animation:
        u_log = model_1.u_log
        return exc_log_1, end_log_1, T_log_1, u_log
    else:
        return exc_log_1, end_log_1, T_log_1


args = [
        (int(10**6), 10**-7, 50, 50,  (1/3)*(0.5**3)*(10**-2)),
        (int(10**6), 10**-7, 315, 315,  (1/3)*(0.5**3)*(10**-2)),
        ]

print(f"Start time {datetime.datetime.now()}")

with ThreadPoolExecutor(max_workers = 2) as executor:
    results = list(executor.map(run_single_simulation, args))

# for animation 
if animation:
    list_a, list_b, list_c, u_log = zip(*results)
    pickle.dump(u_log, open('u_log.pkl', 'wb'))

#pickle.dump(results, open('end.pkl', 'wb'))


