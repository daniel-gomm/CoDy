import numpy as np
from cody.constants import COL_ID, COL_NODE_U, COL_NODE_I

def k_hop_temporal_subgraph(df, num_hops, event_idx):
    """
    df: temporal graph, events stream. DataFrame. An user-item bipartite graph.
    node: center user node
    num_hops: number of hops of the subgraph
    event_idx: should start from 1. 1, 2, 3, ...
    return: a sub DataFrame

    """
    df_new = df.copy()
    df_new = df_new[df_new[COL_ID] <= event_idx]  # ignore events later than event_idx

    center_node = df_new[df_new[COL_ID] == event_idx][COL_NODE_U].values[0]  # event_idx represents e_idx

    subsets = [[center_node], ]
    num_nodes = np.max((df_new[COL_NODE_I].max(), df_new[COL_NODE_U].max())) + 1

    node_mask = np.zeros((num_nodes,), dtype=bool)
    source_nodes = np.array(df_new[COL_NODE_U], dtype=int)  # user nodes, 0--k-1
    target_nodes = np.array(df_new[COL_NODE_I],
                            dtype=int)  # item nodes, k--N-1, N is the number of total users and items

    for _ in range(num_hops):
        node_mask.fill(False)
        node_mask[np.array(subsets[-1])] = True
        edge_mask = node_mask[source_nodes]
        new_nodes = target_nodes[edge_mask]  # new neighbors
        subsets.append(np.unique(new_nodes).tolist())

        source_nodes, target_nodes = target_nodes, source_nodes  # regarded as undirected graph

    subset = np.unique(np.concatenate([np.array(nodes) for nodes in subsets]))  # selected temporal subgraph nodes

    assert center_node in subset

    source_nodes = np.array(df_new[COL_NODE_U], dtype=int)
    target_nodes = np.array(df_new[COL_NODE_I], dtype=int)

    node_mask.fill(False)
    node_mask[subset] = True

    user_mask = node_mask[source_nodes]  # user mask for events
    item_mask = node_mask[target_nodes]  # item mask for events

    edge_mask = user_mask & item_mask  # event mask

    subgraph_df = df_new.iloc[edge_mask, :].copy()
    assert center_node in subgraph_df[COL_NODE_U].values

    return subgraph_df

def greedy_highest_value_over_array(values):
    best_values = [values[0], ]
    best = values[0]
    for i in range(1, len(values)):
        if best < values[i]:
            best = values[i]
        best_values.append(best)
    return np.array(best_values)
