from concurrent.futures import ThreadPoolExecutor
from cpp import DynamicModelDis
from cpp import DynamicModelForce
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

edges,number_of_nodes = read_mesh_to_edges("kamerton_mesh.msh")
BC_nodes = get_nodes_for_line("kamerton_mesh.msh", "BC")
end_nodes = get_nodes_for_line("kamerton_mesh.msh", "end")
nodes = read_mesh_pos_as_dofs("kamerton_mesh.msh")

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




def run_single_simulationForce(args):
    steps,dt, strart_f, end_f, damping_div,animation = args
    print(f"Running simulation with start_f: {strart_f}, end_f: {end_f}, damping_div: {damping_div}")
    model_1 = DynamicModelForce.DynamicModelForce(triplets,nrows,ncols,BC_nodes,end_nodes,number_of_nodes, damping_div)
    print(model_1.mass_per_dof)
    exc_log_1, end_log_1, T_log_1 = model_1.run_simulation(steps, dt, strart_f, end_f, 100,animation)
    
    if animation:
        u_log = model_1.u_log
        return exc_log_1, end_log_1, T_log_1, u_log
    else:
        return exc_log_1, end_log_1, T_log_1

def run_single_simulationDis(args):
    steps,dt, strart_f, end_f, damping_div,animation = args
    print(f"Running simulation with start_f: {strart_f}, end_f: {end_f}, damping_div: {damping_div}")
    model_1 = DynamicModelDis.DynamicModelDis(triplets,nrows,ncols,BC_nodes,end_nodes,number_of_nodes, damping_div)
    print(model_1.mass_per_dof)
    exc_log_1, end_log_1, T_log_1 = model_1.run_simulation(steps, dt, strart_f, end_f, 100, animation)
    
    if animation:
        u_log = model_1.u_log
        return exc_log_1, end_log_1, T_log_1, u_log
    else:
        return exc_log_1, end_log_1, T_log_1

# 56 moda z ugieciem na mocowaniu
# 90 moda kamertonu
# tlumienie dla sil (1/3)*(0.5**3)*(10**-2)
# tlumienie dla dis (1/1)*(0.5**3)*(10**-2)
"""
anim = True
args_force = [
        (int(10**6), 10**-7, 56, 56,  (1/3)*(0.5**3)*(10**-2), anim),
        (int(10**6), 10**-7, 90, 90,  (1/3)*(0.5**3)*(10**-2), anim),
        ]

args_dis = [
        (int(10**6), 10**-7, 56, 56,  (1/1)*(0.5**3)*(10**-2), anim),
        (int(10**6), 10**-7, 90, 90,  (1/1)*(0.5**3)*(10**-2), anim),
        ]

with ThreadPoolExecutor(max_workers = 2) as executor:
    results_force = list(executor.map(run_single_simulationForce, args_force))

with ThreadPoolExecutor(max_workers = 2) as executor:
    results_dis = list(executor.map(run_single_simulationDis, args_dis))
# for animation 
if anim:
    list_a, list_b, list_c, u_log = zip(*results_force)
    pickle.dump(u_log, open('anim_force.pkl', 'wb'))

# for animation 
if anim:
    list_a, list_b, list_c, u_log = zip(*results_dis)
    pickle.dump(u_log, open('anim_dis.pkl', 'wb'))
"""

args_force = [
        (int(3*10**5), 10**-7, 30, 120,  (1/3)*(0.5**3)*(10**-2), False),
        ]
args_dis = [
        (int(5*10**6), (3/5)*10**-7, 120, 120,  (1/1)*(0.5**3)*(10**-2), False),
        ]

#with ThreadPoolExecutor(max_workers = 2) as executor:
#    results_force = list(executor.map(run_single_simulationForce, args_force))
#pickle.dump(results_force, open('end_force.pkl', 'wb'))

with ThreadPoolExecutor(max_workers = 2) as executor:
    results_dis = list(executor.map(run_single_simulationDis, args_dis))
pickle.dump(results_dis, open('end_dis_2.pkl', 'wb'))


