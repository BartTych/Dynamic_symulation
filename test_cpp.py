import basic
from cpp import s_loader
from import_mesh import read_mesh_to_edges
from import_mesh import get_nodes_for_line
from asseble_stiffness_matrix import assemble_global_stiffness
import numpy as np

edges,number_of_nodes = read_mesh_to_edges("test_5.msh")
BC_nodes = get_nodes_for_line("test_5.msh", "BC")
end_nodes = get_nodes_for_line("test_5.msh", "end")

print(f"excitation nodes: {BC_nodes}")
print(f"end nodes: {end_nodes}")

print('start assembling stiffness matrix')
K = assemble_global_stiffness(edges, number_of_nodes)
print('finished assembling stiffness matrix')

rows, cols = K.nonzero()
values = K.data
triplets = np.vstack((rows, cols, values)).T.astype(np.float64) 

print('assembling sparse matrix using cpp')
A = s_loader.build_sparse_from_triplets(triplets,rows.max() + 1, cols.max() + 1)
print('finished assembling sparse matrix using cpp !!!')