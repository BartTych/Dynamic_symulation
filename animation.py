import numpy as np
import pickle
from matplotlib import pyplot as plt

from import_mesh import read_mesh_pos_as_dofs


import matplotlib.animation as animation
from matplotlib import rcParams


u_log = pickle.load(open('u_log.pkl', 'rb'))
u_log = u_log[0]
nodes = read_mesh_pos_as_dofs("test_5.msh")


def position_ulog(u_log, points):
    xy_flat = points[:, :2].flatten().tolist()  # base node XY coordinates
    positioned_ulog = [
        [ui + xi for ui, xi in zip(u_frame, xy_flat)]
        for u_frame in u_log
    ]
    return positioned_ulog


def separtate_ulog_x_and_y(u_log):
    new_log = []
    for n in u_log:
        xs = n[::2]  # Every 2nd element starting at index 0 (x values)
        ys = n[1::2] # Every 2nd element starting at index 1 (y values)
        new_log.append((xs,ys))
    return new_log


def prepare_animation(u_log, nodes, frame_stride=1, point_stride=1, y_multiplication = 10):
    # Convert displacements to absolute positions
    u_log = position_ulog(u_log, nodes)
    
    # Separate into x and y lists per frame: [(xs0, ys0), (xs1, ys1), ...]
    u_log = separtate_ulog_x_and_y(u_log)

    # Apply frame stride
    u_log = u_log[::frame_stride]

    # Apply point stride inside each (xs, ys)
    u_log = [
        (xs[::point_stride], ys[::point_stride])
        for xs, ys in u_log
    ]

    u_log = [
        (xs, [y * y_multiplication for y in ys])
        for xs, ys in u_log
    ]

    return u_log

def prepare_animation_chat(u_log, nodes, frame_stride=1, point_stride=1, y_multiplication=10):
    xy_flat = nodes[:, :2].flatten().tolist()  # base x0,y0,x1,y1,...

    u_log_prepared = []

    for frame_i, u_frame in enumerate(u_log[::frame_stride]):
        if len(u_frame) != len(xy_flat):
            raise ValueError(f"Displacement length mismatch at frame {frame_i}: {len(u_frame)} vs {len(xy_flat)}")

        # Scale displacement in Y only (odd indices)
        u_frame_scaled = [
            uy * y_multiplication if i % 2 == 1 else ux
            for i, (ux, uy) in enumerate(zip(u_frame[::2], u_frame[1::2]))
            for _ in (0, 1)
        ]
        # Reconstruct full scaled displacement vector: [dx0, dy0, dx1, dy1, ...]
        u_scaled_flat = []
        for i in range(0, len(u_frame), 2):
            u_scaled_flat.append(u_frame[i])             # dx
            u_scaled_flat.append(u_frame[i+1] * y_multiplication)  # dy scaled

        # Compute absolute position
        abs_pos = [u + x for u, x in zip(u_scaled_flat, xy_flat)]

        # Split and stride
        xs = abs_pos[::2][::point_stride]
        ys = abs_pos[1::2][::point_stride]

        u_log_prepared.append((xs, ys))

    return u_log_prepared



def animate_displacement(u_log, interval=50, point_size=2, save_path=None):
    """
    u_log: list of (xs, ys) tuples
    interval: milliseconds between frames
    point_size: scatter point size
    save_path: if set (e.g., 'animation.gif'), saves to that path
    """
    fig, ax = plt.subplots()
    xs, ys = u_log[0]
    scat = ax.scatter(xs, ys, s=point_size)

    # Set limits
    all_x = [x for frame in u_log for x in frame[0]]
    all_y = [y for frame in u_log for y in frame[1]]
    ax.set_xlim(min(all_x), max(all_x))
    ax.set_ylim(min(all_y), max(all_y))
    ax.set_aspect('equal')

    def update(i):
        x, y = u_log[i]
        data = np.stack([x, y], axis=1)  # or .T if stacking on axis=0
        scat.set_offsets(data)
        return scat,


    ani = animation.FuncAnimation(
        fig, update, frames=len(u_log), interval=interval, blit=True
    )

    
    ani.save(save_path, writer='pillow', fps=1000 // interval)

u_log = prepare_animation_chat(u_log,nodes,frame_stride=1,point_stride=10,y_multiplication=5)    
#u_log = prepare_animation(u_log, nodes,frame_stride = 100, point_stride = 100,y_multiplication = 1000)
animate_displacement(u_log,save_path='animation.gif')

def plot_frame(u_log, frame_index=0, y_multiplication=1, point_size=2):
    """
    Plot the n-th frame of u_log after processing.
    
    u_log: list of (xs, ys)
    frame_index: index of the frame to plot
    y_multiplication: scale factor applied to Y values
    point_size: scatter marker size
    """
    if frame_index >= len(u_log):
        raise IndexError(f"Frame index {frame_index} out of range. u_log has {len(u_log)} frames.")
    
    xs, ys = u_log[frame_index]
    xs = np.array(xs)
    ys = np.array(ys) * y_multiplication

    fig, ax = plt.subplots()
    ax.scatter(xs, ys, s=point_size)
    ax.set_aspect('equal')
    ax.set_title(f"Frame {frame_index}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y (scaled)")
    plt.show()


