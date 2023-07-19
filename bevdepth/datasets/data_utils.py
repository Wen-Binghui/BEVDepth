import numpy as np

def depthImg2scatter(dep_np: np.ndarray) -> np.ndarray:
    # 
    dim = dep_np.shape
    X, Y = np.meshgrid(np.arange(dim[0]), np.arange(dim[1]), indexing='ij')
    X = np.reshape(X, (-1,1))
    Y = np.reshape(Y, (-1,1))
    dep_np = np.reshape(dep_np, (-1,1))
    selected_ind = dep_np>0
    point_set = np.vstack([X[selected_ind], Y[selected_ind], dep_np[selected_ind]])


    return point_set

# def lidar_point_cutoff()