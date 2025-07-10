import numpy as np
import pickle
from matplotlib import pyplot as plt

from import_mesh import read_mesh_pos_as_dofs
from animation_forlder import animation_prep

u_log_data = pickle.load(open('anim_force.pkl', 'rb'))

nodes = read_mesh_pos_as_dofs("kamerton_mesh.msh")

for i, u_log in enumerate(u_log_data):
    
    
    u_log = animation_prep.prepare_animation(u_log,nodes, frame_stride=2, point_stride=3, y_multiplication=1)    
    #u_log = prepare_animation(u_log, nodes,frame_stride = 100, point_stride = 100,y_multiplication = 1000)
    animation_prep.animate_displacement(u_log,save_path=f"animation{i}.gif")


