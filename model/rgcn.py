import torch
import torch.nn.functional as F
from torch import nn

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, dropout=0.2):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.activation = nn.ReLU()

        # Relation-specific weight matrices
        self.weight = nn.Parameter(torch.Tensor(num_rels, in_feat, out_feat))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # Self-loop weight
        self.self_loop = nn.Parameter(torch.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.self_loop, gain=nn.init.calculate_gain('relu'))

    def forward(self, node_features, adj_list, rel_type, training=True, drop_prob=0.2):
        # node_features: [N, B, D] → [B, N, D]
        node_features = node_features.permute(1, 0, 2)

        output = torch.zeros_like(node_features)

        for r in range(self.num_rels):
            mask = (rel_type == r).float()  # [B, N, N]
            weight_r = self.weight[r]       # [D, D]
            neighbor_count = torch.clamp(torch.sum(adj_list, dim=-1, keepdim=True), min=1)

            weighted_sum = torch.matmul(node_features, weight_r)        # [B, N, D]
            masked_adj_list = adj_list * mask                           # [B, N, N]
            weighted_sum = torch.matmul(masked_adj_list, weighted_sum)  # [B, N, D]

            output += weighted_sum / neighbor_count

        output += torch.matmul(node_features, self.self_loop)
        output = self.activation(output)

        return output.permute(1, 0, 2)  # [N, B, D]


class RelationalGCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_rels, rgcn_layers_num=2):
        super().__init__()

        assert input_dim == output_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.encoding_layer = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.decoding_layer = nn.Parameter(torch.Tensor(hidden_dim, output_dim))
        nn.init.xavier_uniform_(self.encoding_layer, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.decoding_layer, gain=nn.init.calculate_gain('relu'))

        self.rgcn_layers = nn.ModuleList([
            RGCNLayer(hidden_dim, hidden_dim, num_rels) for _ in range(rgcn_layers_num)
        ])

    def forward(self, node_features, adj_list, rel_type, training=True, drop_prob=0.2):
        # node_features: [N, B, D]
        encoded_node_features = torch.matmul(node_features, self.encoding_layer)  # [N, B, H]

        start_encoded_node_features = encoded_node_features

        print(f"before rgcn: start_encoded_node_features[0][0]: {start_encoded_node_features[2][0]}")


        for rgcn in self.rgcn_layers:
            prev_features = encoded_node_features
            encoded_node_features = rgcn(
                encoded_node_features,
                adj_list,
                rel_type,
                training=training,
                drop_prob=drop_prob
            )
            encoded_node_features = encoded_node_features + prev_features  # residual

        final_encoded_node_features = encoded_node_features
        print(f"after rgcn: final_encoded_node_features[0][0]: {final_encoded_node_features[2][0]}")

        print(f"change after rgcn model: {final_encoded_node_features[2][0] - start_encoded_node_features[2][0]}")


        node_features = torch.matmul(encoded_node_features, self.decoding_layer)
        return node_features  # [N, B, D]
