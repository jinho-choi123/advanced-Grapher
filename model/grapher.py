import torch
import torch.nn.functional as F
from torch import nn

class Grapher(nn.Module):
    def __init__(self,
                 transformer_class,
                 transformer_name,
                 cache_dir,
                 max_nodes,
                 edges_as_classes,
                 node_sep_id,
                 default_seq_len_edge,
                 num_classes,
                 dropout_rate,
                 num_layers,
                 vocab_size,
                 bos_token_id,
                        ):
        super().__init__()

        self.transformer = transformer_class.from_pretrained(transformer_name, cache_dir=cache_dir)

        self.hidden_dim = self.transformer.config.d_model
        self.max_nodes = max_nodes
        self.edges_as_classes = edges_as_classes
        self.node_sep_id = node_sep_id
        self.default_seq_len_edge = default_seq_len_edge

        # always use edgesclass
        self.edges = EdgesClass(self.hidden_dim, num_classes, dropout_rate, num_layers)

    def split_nodes(self, output_ids, features):

        # features: batch_size x seq_len x hidden_dim
        # output_ids: batch_size x seq_len

        batch_size, _ = output_ids.size()
        split_features = torch.zeros((self.max_nodes, batch_size, self.hidden_dim), device=features.device, dtype=features.dtype)  # num_nodes X batch_size X hidden_dim

        for n in range(self.max_nodes):
            mask_node_n = ((torch.cumsum((output_ids == self.node_sep_id), 1) == n) & (output_ids != self.node_sep_id)).unsqueeze(2)
            features_node_n = features*mask_node_n
            sum_features_node_n = torch.cumsum(features_node_n, 1)[:, -1]
            num_tokens_node_n = torch.sum(mask_node_n, 1)
            num_tokens_node_n[num_tokens_node_n == 0] = 1
            ave_features_node_n = sum_features_node_n / num_tokens_node_n
            split_features[n] = ave_features_node_n

        return split_features

    def forward(self, text, text_mask, target_nodes, target_nodes_mask, target_edges, output_hidden_states=False):
        # NODES
        output = self.transformer(input_ids=text,
                                attention_mask=text_mask,
                                decoder_input_ids=target_nodes,
                                decoder_attention_mask=target_nodes_mask,
                                output_hidden_states=True)

        logits_nodes = output.logits  # batch_size x seq_len x vocab_size
        joint_features = output.decoder_hidden_states[-1]  # batch_size x seq_len x hidden_dim

        gen_seq = logits_nodes.argmax(-1)

        # num_nodes x batch_size x hidden_dim
        features = self.split_nodes(gen_seq, joint_features)

        # if return_hidden is set, then we return hidden embedding of node
        # This is because, we can easily derive hidden embedding of node-node connection. Please look at EdgeClass class for more infromation.


        # EDGES
        logits_edges = self.edges(features)


        if output_hidden_states:
            return logits_nodes, logits_edges, features
        else:
            return logits_nodes, logits_edges

    def sample(self, text, text_mask):

        # NODES
        output = self.transformer.generate(input_ids=text,
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

        seq_len_edge = self.default_seq_len_edge

        # num_nodes x batch_size x hidden_dim
        features = self.split_nodes(seq_nodes, joint_features)

        # EDGES
        # num_nodes x num_nodes x batch_size x class_num
        logits_edges = self.edges(features)

        seq_edges = logits_edges.argmax(-1)

        return logits_nodes, seq_nodes, logits_edges, seq_edges


class EdgesClass(nn.Module):
    def __init__(self, hidden_dim, num_classes, dropout_rate=0.5, num_layers=0):
        super(EdgesClass, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.layers = nn.Sequential()

        dim = num_classes
        self.layers.add_module('first', nn.Linear(hidden_dim, dim))
        self.layers.add_module('firstrelu', nn.ReLU())
        self.layers.add_module('firstdropout', nn.Dropout(dropout_rate))
        for l in range(num_layers):
            self.layers.add_module(f'lin{l}', nn.Linear(dim, dim))
            self.layers.add_module(f'relu{l}', nn.ReLU())
            self.layers.add_module(f'dropout{l}', nn.Dropout(dropout_rate))
        self.layers.add_module('last', nn.Linear(dim, num_classes))

    def forward(self, features):

        # features: num_nodes X batch_size X hidden_dim

        num_nodes = features.size(0)
        batch_size = features.size(1)

        # num_nodes_valid X num_nodes_valid X batch_size X hidden_dim
        feats = features.unsqueeze(0).expand(num_nodes, -1, -1, -1)

        # [featurs[i] - features[j]]: (num_nodes_valid*num_nodes_valid*batch_size) X hidden_dim
        hidden = (feats.permute(1, 0, 2, 3) - feats).reshape(-1, self.hidden_dim)

        # logits: num_nodes_valid*num_nodes_valid*batch_size X num_classes
        logits = self.layers(hidden)

        # num_nodes X num_nodes X batch_size X num_classes
        all_logits = logits.reshape(num_nodes, num_nodes, batch_size, -1)

        return all_logits
