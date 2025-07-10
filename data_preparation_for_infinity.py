
from concurrent.futures import ThreadPoolExecutor
from cpp import DynamicModelDis_inf

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


def run_single_simulationDis_inf(args):
    steps,dt, strart_f, end_f, damping_div,anim = args
    print(f"Running simulation with start_f: {strart_f}, end_f: {end_f}, steps: {steps}")
    model_1 = DynamicModelDis_inf.DynamicModelDis_inf(triplets,nrows,ncols,BC_nodes,end_nodes,number_of_nodes, damping_div)
    exc_log, end_log_x, end_log_y = model_1.run_simulation(steps, dt, strart_f, end_f, 100, anim)
    
    return exc_log, end_log_x, end_log_y
    
# 56 moda z ugieciem na mocowaniu
# 90 moda kamertonu
# tlumienie dla sil (1/3)*(0.5**3)*(10**-2)
# tlumienie dla dis (1/1)*(0.5**3)*(10**-2)

anim = False

# tutaj zrobie petle generujaca przypadki
length = 0.05 # [s]

steps = [3 * 10**5 + 20_000 * n for n in range(135)]

print(steps)
args_dis = [(int(n), length/n, 120, 120,  (1/1)*(0.5**3)*(10**-2), anim) for n in steps]

with ThreadPoolExecutor(max_workers = 3) as executor:
    results_dis = list(executor.map(run_single_simulationDis_inf, args_dis))
pickle.dump(results_dis, open('end_inf.pkl', 'wb'))


