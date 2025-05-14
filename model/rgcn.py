import torch
import torch.nn.functional as F
from torch import nn

class RelationalGCN(nn.Module):
    def __init__(self, grapher):
        super().__init__()

        self.grapher = grapher

    def forward(self, batch, batch_idx):
        # target_nodes: batch_size X seq_len_node
        # target_edges: num_nodes X num_nodes X batch_size X seq_len_edge [FULL]
        # target_edges: batch_size X num_nodes X num_nodes [CLASSES]
        text_input_ids, text_input_attn_mask, target_nodes, target_nodes_mask, target_edges = batch

        # Extract node's feature embedding
        node_features = self.grapher(text_input_ids,
                                               text_input_attn_mask,
                                               target_nodes,
                                               target_nodes_mask,
                                               target_edges, output_hidden_states=True)
