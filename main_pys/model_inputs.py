import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
import numpy as np

def create_data_object(pos_list, bd_list, grid, k, m, goal_locs, labels=np.array([]), debug_checks=False):
    """
    pos_list: (N,2) positions
    bd_list: (N,W,H) bd's
    grid: (W,H) grid
    k: (int) local region size
    m: (int) number of closest neighbors to consider
    """
    num_layers = 3 # grid and bd_slices intially
    
        
    num_agents = len(pos_list)
    range_num_agents = np.arange(num_agents)

    ### Numpy advanced indexing to get all agent slices at once
    rowLocs = pos_list[:,0][:, None] # (N)->(N,1), Note doing (N)[:,None] adds an extra dimension
    colLocs = pos_list[:,1][:, None] # (N)->(N,1)
    if debug_checks:
        assert(grid[pos_list[:,0], pos_list[:,1]].all() == 0) # Make sure all agents are on empty space

    x_mesh, y_mesh = np.meshgrid(np.arange(-k,k+1), np.arange(-k,k+1), indexing='ij') # Each is (D,D)
    # Adjust indices to gather slices
    x_mesh = x_mesh[None, :, :] + rowLocs[:, None, :] # (1,D,D) + (D,1,D) -> (N,D,D)
    y_mesh = y_mesh[None, :, :] + colLocs[:, None, :] # (1,D,D) + (D,1,D) -> (N,D,D)
    grid_slices = grid[x_mesh, y_mesh] # (N,D,D)
    bd_slices = bd_list[range_num_agents[:,None,None], x_mesh, y_mesh] # (N,D,D)
    N,D = bd_slices.shape[0], bd_slices.shape[1]
    node_features = np.empty((N,num_layers,D,D),dtype=np.float32)
    node_feature_idx = 0
    node_features[:,node_feature_idx] = grid_slices
    node_feature_idx +=1
    node_features[:,node_feature_idx] = bd_slices
    node_feature_idx +=1
    goalRowLocs, goalColLocs= goal_locs[:,0][:, None], goal_locs[:,1][:, None]  # (N,1), (N,1)
    matches = (rowLocs == goalRowLocs) & (colLocs == goalColLocs)
    
    # agent positions
    agent_pos = np.zeros((grid.shape[0], grid.shape[1])) # (W,H)
    agent_pos[rowLocs, colLocs] = 1 # (W,H)
    agent_pos_slices = agent_pos[x_mesh, y_mesh] # (N,D,D)
    node_features[:,node_feature_idx] = agent_pos_slices
    node_feature_idx +=1

    agent_indices = np.repeat(np.arange(num_agents)[None,:], axis=0, repeats=m).T # (N,N), each row is 0->num_agents
    deltas = pos_list[:, None, :] - pos_list[None, :, :] # (N,1,2) - (1,N,2) -> (N,N,2), the difference between each agent

    ## Calculate the distance between each agent, einsum is faster than other options
    dists = np.einsum('ijk,ijk->ij', deltas, deltas, optimize='optimal').astype(float) # (N,N), the L2^2 distance between each agent
    # dists2 = np.linalg.norm(deltas, axis=2, ord=2) # (N,N), the distance between each agent
    # dists3 = np.sum(np.abs(deltas)**2, axis=2) # (N,N), the distance between each agent
    # assert(np.allclose(dists1, dists3)) # Make sure the two distance calculations are the same
    # assert(np.allclose(dists1, np.sqrt(dists2))) # Make sure the two distance calculations are the same

    fov_dist = np.any(np.abs(deltas) > k, axis=2) # (N,N,2)->(N,N) bool for if the agent is within the field of view
    dists[fov_dist] = np.inf # Set the distance to infinity if the agent is out of the field of view
    closest_neighbors = np.argsort(dists, axis=1, kind="quicksort")[:, 1:m+1] # (N,m), the indices of the 4 closest agents, ignore self
    # arg_dists = np.argpartition(dists, m+1, axis=1)
    # closest_neighbors = arg_dists[:,1:m+1]
    distance_of_neighbors = dists[range_num_agents[:,None],closest_neighbors] # (N,m)
    
    neighbors_and_source_idx = np.stack([agent_indices, closest_neighbors]) # (2,N,m), 0 stores source agent, 1 stores neigbhor
    selection = distance_of_neighbors != np.inf # (N,m)
    edge_indices = neighbors_and_source_idx[:, selection] # (2, num_edges), [:,i] corresponds to (source, neighbor)
    edge_features = deltas[edge_indices[0], edge_indices[1]] # (num_edges,2), the difference between each agent
    edge_features = edge_features.astype(np.float32)

    if debug_checks:
        assert(node_features[:,0,k,k].all() == 0) # Make sure all agents are on empty space
        
    bd_pred_arr = None
    linear_dimensions = (grid_slices.shape[1]-2)**2 * num_layers
    # TODO get the best location to go next, just according to the bd
    # NOTE: because we pad all bds with a large number, 
    # we should be able to get the up, down, left and right of each bd without fear of invalid indexing
    # (N, [Stop, Right, Down, Up, Left])
    x_mesh2, y_mesh2 = np.meshgrid(np.arange(-1,1+1), np.arange(-1,1+1), indexing='ij') # assumes k at least 1; getting a 3x3 grid centered at the same place
    x_mesh2 = x_mesh2[None, :, :] + rowLocs[:, None, :] #  -> (N,3,3)
    y_mesh2 = y_mesh2[None, :, :] + colLocs[:, None, :] # -> (N,3,3)
    bd_list = bd_list[np.arange(num_agents)[:,None,None], x_mesh2, y_mesh2] # (N,3,3)
    # set diagonal entries to a big number
    flattened = np.reshape(bd_list, (-1, 9)) # (N,9) # (order (top to bot) left mid right, left mid right, left mid right)
    flattened = flattened[:,[(4,5,7,1,3)]].reshape((-1,5)) # (N,5)

    # Create a boolean array where each element is True if it is the minimum in its row
    min_indices = flattened == flattened.min(axis=1, keepdims=True)
    min_indices = min_indices.astype(int) # (N, 5) non-unique argmin solution
    bd_pred_arr = min_indices 
    linear_dimensions+=5
    
    return Data(x=torch.from_numpy(node_features), edge_index=torch.from_numpy(edge_indices), 
                edge_attr=torch.from_numpy(edge_features), bd_pred=torch.from_numpy(bd_pred_arr), lin_dim=linear_dimensions, num_channels=num_layers,
                y = torch.from_numpy(labels))
    


def normalize_graph_data(data, k, edge_normalize="k", bd_normalize="center"):
    """Modifies data in place"""
    ### Normalize edge attributes
    # data.edge_attr (num_edges,2) the deltas in each direction which can be negative
    assert(edge_normalize in ["k"])
    if edge_normalize == "k":
        data.edge_attr /= k # Normalize edge attributes
    else:
        raise KeyError("Invalid edge normalization method: {}".format(edge_normalize))

    ### Normalize bd
    assert(bd_normalize in ["center"])
    bd_grid = data.x # (N,2,D,D)
    center = bd_grid[:, 1, k, k].unsqueeze(1).unsqueeze(2) # (N,1,1)
    bd_grid[:, 1, :, :] -= center
    bd_grid[:, 1, :, :] *= (1 - bd_grid[:, 0, :, :])
    bd_grid[:, 1, :, :] /= (2*k)
    bd_grid[:, 1, :, :] = torch.clamp(bd_grid[:, 1, :, :], min=-1.0, max=1.0)

    data.x = bd_grid
    assert(data.x[:,1,k,k].all() == 0) # Make sure all agents are on empty space
    assert(data.x[:,1].max() <= 1.0 and data.x[:,1].min() >= -1.0) # Make sure all agents are on empty space
    assert(data.x[:,0,k,k].all() == 0) # Make sure all agents are on empty space
    return data