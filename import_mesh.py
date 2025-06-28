import meshio
from itertools import combinations
import math
import numpy as np

def read_mesh_to_edges(mesh_file):
    """
    Reads a mesh file and extracts edges with their angles and lengths. 
    plus returns the number of nodes.
    
    Parameters:
    - mesh_file: Path to the mesh file (e.g., .msh, .vtk)
    
    Returns:
    - edges: List of tuples containing edge nodes, angle, and length
    """
    # Read the mesh file
    mesh = meshio.read(mesh_file)
    #than gives id number of the line BC
    # Extract points and cells
    points = mesh.points  # (N, 3) array
    cells = mesh.cells_dict
    
    # For 2D mesh: use 'triangle', for 3D: use 'tetra'
    elements = cells.get("triangle")
    
    # Create edges from elements
    edges = set()
    for elem in elements:
        for i, j in combinations(elem, 2):
            edge = tuple(sorted((i, j)))
            edges.add(edge)
    
    edges = list(edges)
    
    return compute_edge_angles_and_lengths(edges, points), len(points)

def read_mesh_pos_as_dofs(mesh_file):
    
    mesh = meshio.read(mesh_file)
    points = mesh.points  # (N, 3) array

    return points

def compute_edge_angles_and_lengths(edge_list, points):
    result = []
    for n1, n2 in edge_list:
        # Get 2D coordinates (assume z = 0 or not needed)
        x1, y1 = points[n1][:2]
        x2, y2 = points[n2][:2]
        
        dx = x2 - x1
        dy = y2 - y1
        
        length = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        
        result.append(((n1, n2), angle, length))
    return result


def get_nodes_for_line(mesh_file, name):

    mesh = meshio.read(mesh_file)
    #than gives id number of the line BC
    bc_tag_id = mesh.field_data[name][0]  # mesh.field_data["BC"] = [tag_id, dim]
    #print("Boundary condition tag ID:", bc_tag_id)
    #all line elements in the mesh, lines are on the egdes of the mesh 
    lines_of_mesh = mesh.cells_dict["line"]
    #print(len(lines_of_mesh), "lines in mesh")

    line_tags = mesh.cell_data_dict["gmsh:physical"]["line"]
    #print(line_tags)

    bc_line_nodes = lines_of_mesh[line_tags == bc_tag_id]
    
    return np.unique(bc_line_nodes)