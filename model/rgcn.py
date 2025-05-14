import torch
import torch.nn.functional as F
from torch import nn

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, activation=F.relu):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.activation = activation

        # Relation-specific weight matrices
        self.weight = nn.Parameter(torch.Tensor(num_rels, in_feat, out_feat))
        nn.init.xavier_uniform_(self.weight)

        # Self-loop weight
        self.self_loop = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.self_loop)

    def forward(self, node_features, adj_list, rel_type):
        # convert node_features from num_nodes x batch_size x hidden
        # to batch_size x num_nodes x hidden
        node_features = node_features.permute(1, 0, 2)

        output = torch.zeros_like(node_features)

        for r in range(self.num_rels):
            mask = (rel_type == r).float()
            weight_r = self.weight[r]

            neighbor_count = torch.clamp(torch.sum(adj_list, dim=-1, keepdim=True), min=1)

            # DEBUG
            tmp1 = adj_list * mask # batch_size x node_nums x node_nums
            tmp2 = torch.matmul(node_features, weight_r) #
            tmp3 = torch.matmul(tmp1, tmp2)

            weighted_sum = torch.matmul(adj_list * mask, torch.matmul(node_features, weight_r))

            output += weighted_sum / neighbor_count

        output += torch.matmul(node_features, self.self_loop)

        output = self.activation(output)

        output = output.permute(1, 0, 2)

        return output

class RelationalGCN(nn.Module):
    def __init__(self, hidden_dim, num_rels, rgcn_layers_num=2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.rgcn_layers = nn.ModuleList([
            RGCNLayer(hidden_dim, hidden_dim, num_rels) for _ in range(rgcn_layers_num)
        ])

    def forward(self, node_features, adj_list, rel_type):
        # node_features shape: node_num x batch_size x hidden_dim
        # adj_list shape: batch_size x node_num x node_num
        for rgcn in self.rgcn_layers:
            node_features = rgcn(node_features, adj_list, rel_type)

        return node_features
