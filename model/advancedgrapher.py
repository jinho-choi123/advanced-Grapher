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
        # features: num_nodes x batch_size x hidden_dim
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

    def sample(self, text, text_mask):

        output = self.grapher.transformer.generate(input_ids=text,
                                           max_length=150,
                                           attention_mask=text_mask,
                                           output_hidden_states=True,
                                           output_scores=True,
                                           return_dict_in_generate=True)
        seq_nodes = output.sequences[:, 1:]

        logits_nodes = output.scores  # [batch_size x vocab_size] of length seq_len

        # batch_size x seq_len x vocab_size
        logits_nodes = torch.cat([l.unsqueeze(0) for l in logits_nodes], 0).permute(1, 0, 2)

        # batch_size x seq_len x hidden_dim
        joint_features = torch.cat([h[-1] for h in output.decoder_hidden_states], 1)

        seq_len_edge = self.grapher.default_seq_len_edge

        # num_nodes x batch_size x hidden_dim
        features = self.grapher.split_nodes(seq_nodes, joint_features)


        # EDGES
        logits_edges = self.grapher.edges(features)

        # draw initial graph with logits_edges
        # rel_type: batch_size x num_nodes x num_nodes
        # adj_list: batch_size x num_nodes x num_nodes
        rel_type = logits_edges.permute(2, 0, 1, 3).argmax(-1)

        adj_list = (rel_type != self.noedge_cl)

        # apply rgcn
        features = self.rgcn(features, adj_list, rel_type)

        # generate edges
        logits_edges = self.grapher.edges(features)

        seq_edges = logits_edges.argmax(-1)

        return logits_nodes, seq_nodes, logits_edges, seq_edges
