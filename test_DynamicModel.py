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
model = DynamicModel.DynamicModel(
    triplets,
    nrows,
    ncols,
    BC_nodes,
    end_nodes,
    number_of_nodes)
print('finished creation of DynamicModel instance !!!')

print(model.fixed_dofs)
print(model.excitation_dofs)
print(model.response_dofs)
print(model.damping_coefficient)
star = datetime.datetime.now()
exc_log, end_log = model.run_simulation(10**6,10**-5, 10)
end = datetime.datetime.now()
print(f"simulation took {end - star} seconds")

plt.plot(exc_log, label='excitation')
plt.plot(end_log, label='end response')
plt.show()
