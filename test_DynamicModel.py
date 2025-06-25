from cpp import DynamicModel
from matplotlib import pyplot as plt
import basic
from cpp import simulation
from import_mesh import read_mesh_to_edges
from import_mesh import get_nodes_for_line
from asseble_stiffness_matrix import assemble_global_stiffness
import numpy as np
import datetime

edges,number_of_nodes = read_mesh_to_edges("test_5.msh")
BC_nodes = get_nodes_for_line("test_5.msh", "BC")
end_nodes = get_nodes_for_line("test_5.msh", "end")

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

print('first instance of DynamicModel')
model_1 = DynamicModel.DynamicModel(triplets,nrows,ncols,BC_nodes,end_nodes,number_of_nodes)
model_2 = DynamicModel.DynamicModel(triplets,nrows,ncols,BC_nodes,end_nodes,number_of_nodes)
model_3 = DynamicModel.DynamicModel(triplets,nrows,ncols,BC_nodes,end_nodes,number_of_nodes)
print('finished creation of DynamicModel instance !!!')


star = datetime.datetime.now()

multiplier = 6
exc_log_1, end_log_1, T_log_1 = model_1.run_simulation(multiplier*10**5,10**-5/multiplier, 10)
multiplier = 4
exc_log_2, end_log_2, T_log_2 = model_2.run_simulation(multiplier*10**5,10**-5/multiplier, 10)
multiplier = 2
exc_log_3, end_log_3, T_log_3 = model_3.run_simulation(multiplier*10**5,10**-5/multiplier, 10)

end = datetime.datetime.now()
print(f"simulation took {end - star} seconds")

plt.plot(T_log_1,end_log_1, label='end response 1')
plt.plot(T_log_2,end_log_2, label='end response 2')
plt.plot(T_log_3,end_log_3, label='end response 3')
plt.show()
