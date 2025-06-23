from import_mesh import read_mesh_to_edges
from import_mesh import get_nodes_for_line
from asseble_stiffness_matrix import assemble_global_stiffness
import numpy as np
from matplotlib import pyplot as plt
import math
import frequency_sweep

edges,number_of_nodes = read_mesh_to_edges("test_5.msh")
BC_nodes = get_nodes_for_line("test_5.msh", "BC")
end_nodes = get_nodes_for_line("test_5.msh", "end")

print(f"number of nodes: {number_of_nodes}")
print(f"BC nodes: {BC_nodes} number of bc nodes: {len(BC_nodes)}")
print(f"end nodes: {end_nodes} number of end nodes: {len(end_nodes)}")

print('start assembling stiffness matrix')
K = assemble_global_stiffness(edges, number_of_nodes)
print('finished assembling stiffness matrix')

m = 0.00066 # mass per node [kg]
m_vec = np.full(number_of_nodes * 2, m)  # or node-specific values
inv_M = 1.0 / m_vec

dt = 0.000001
n_steps = 200_000
omega = 3      # excitation frequency [rad/s]
amplitude = 0.001

damping_ratio = 0.001  # damping ratio
damping_coefficient = 2 * math.sqrt(15_000_000+0.03) * damping_ratio

u = np.zeros(number_of_nodes * 2)  # initial displacement
v = np.zeros(number_of_nodes * 2)  # initial displacement

fixed_dofs = [2 * i for i in BC_nodes]
fixed_dofs += [2 * i + 1 for i in BC_nodes]  # fixed dofs at excitation nodes

excitation_dofs = [2 * i for i in end_nodes]

response_dofs = [2 * i for i in end_nodes]  # response at end nodes

all_dofs = np.arange(number_of_nodes * 2)
free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

exc_log = []
end_log = []



f_ext = np.zeros(number_of_nodes * 2)  # external force vector
force = 10
f_ext[excitation_dofs] = force


for i,step in enumerate(range(n_steps)):
    
    if i < n_steps / 2:
      f_ext[excitation_dofs] = force * (0.5*np.sin((i *4/ n_steps) * np.pi /2 -np.pi/2)+0.5)  # sinusoidal force at excitation nodes
    else:
      f_ext[excitation_dofs] = force 
    T = step * dt
    #u[excitation_dofs] = exc_x[i]
    #v[excitation_dofs] = exc_v[i]
    #u[excitation_dofs] = amplitude * np.sin(omega * T)
    #v[excitation_dofs] = amplitude * omega * np.cos(omega * T)

    #f_ext = np.zeros(number_of_nodes * 2)  # external force vector
    #f_ext[excitation_dofs] = 10 * np.sin(omega * T)

    f_int = K @ u
      # net force vector
    a = np.zeros_like(u)  # acceleration vector
    f_damp = damping_coefficient * v  # damping force
    a[free_dofs] = (-f_damp[free_dofs] - f_int[free_dofs] + f_ext[free_dofs] )* inv_M[free_dofs]
    #a[free_dofs] = ( - f_int[free_dofs] + f_ext[free_dofs] )* inv_M[free_dofs]
    
    #a[free_dofs] = (-f_damp[free_dofs] - f_int[free_dofs] + f_ext[free_dofs])* inv_M[free_dofs]
    v[free_dofs] += a[free_dofs] * dt
    # core of symplectic method, u(n+1) = u(n) + v(n+1) * dt
    u[free_dofs] += v[free_dofs] * dt

    if step % 10 == 0:
        exc_log.append(u[excitation_dofs[0]])
        end_log.append(u[response_dofs[0]])
        print(step)

plt.plot(end_log)
print(f"force acting on excitation dofs: {len(end_nodes)* force}")
#plt.plot(exc_log)
plt.show()