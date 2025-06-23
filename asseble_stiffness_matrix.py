import numpy as np
from scipy.sparse import lil_matrix

def assemble_global_stiffness(edge_data, number_of_nodes, C=100_000_000.0):
    dof = 2
    size = dof * number_of_nodes
    K = lil_matrix((size, size))

    for (i, j), theta, length in edge_data:
        c = np.cos(theta)
        s = np.sin(theta)
        #k = C / length
        # straight stiffness matrix for a 2D beam element
        k = C 

        k_local = k * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])

        dof_map = [2*i, 2*i+1, 2*j, 2*j+1]

        # Block insertion
        K[np.ix_(dof_map, dof_map)] += k_local
        #start = time.perf_counter()
        
        # Convert to CSR format if needed
    print('converting to CSR format')
    K =  K.tocsr()
        #end = time.perf_counter()
    return K
