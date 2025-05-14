from model import grapher, rgcn
import torch
import torch.nn.functional as F
from torch import nn

class AdvancedGrapher(nn.Module):
    def __init__(self, grapher, rgcn, noedge_cl):
        super().__init__()
        self.grapher = grapher
        self.rgcn = rgcn
        self.noedge_cl = noedge_cl

    def forward(self, text, text_mask, target_nodes, target_nodes_mask, target_edges, output_hidden_states=False):
        # logits_edges: num_nodes x num_nodes x batch_size x num_classes
        logits_nodes, logits_edges, features = self.grapher(text, text_mask, target_nodes, target_nodes_mask, target_edges, True)

        # logits_edges: batch_size x num_nodes x num_nodes x num_classes
        logits_edges = logits_edges.permute(2, 0, 1, 3)

        # draw initial graph with logits_edges
        # rel_type: batch_size x num_nodes x num_nodes
        # adj_list: num_nodes x num_nodes
        rel_type = logits_edges.argmax(-1)

        adj_list = (rel_type != self.noedge_cl)

        # apply rgcn
        features = self.rgcn(features, adj_list, rel_type)

        # generate edges
        logits_edges = self.grapher.edges(features)

        return logits_nodes, logits_edges
