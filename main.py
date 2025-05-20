from data.dataset import GraphDataModule
from pytorch_lightning import loggers as pl_loggers
import argparse
from model.litgrapher import LitGrapher
import pytorch_lightning as pl
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from misc.utils import decode_graph
import pprint

class SeparateCheckpoint(ModelCheckpoint):
    def __init__(self, grapher_ckpt_path, rgcn_ckpt_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grapher_ckpt_path = grapher_ckpt_path
        self.rgcn_ckpt_path = rgcn_ckpt_path

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):

        state_dict = checkpoint['state_dict']

        # parameters of the grapher model
        grapher_state_dict = {k.replace('model.grapher.', ''): v for k, v in state_dict.items() if k.startswith('model.grapher.')}
        # parameters of the rgcn model
        rgcn_state_dict = {k.replace('model.rgcn.', ''): v for k, v in state_dict.items() if k.startswith('model.rgcn.')}

        import torch

        # # save the grapher model : if it not exists
        # if not os.path.exists(self.grapher_ckpt_path):
        #     torch.save({'state_dict': grapher_state_dict}, self.grapher_ckpt_path)
        #     print(f"Saved Grapher checkpoint to {self.grapher_ckpt_path}")
        # else:
        #     print(f"Checkpoint already exists at {self.grapher_ckpt_path}, not saving again.")

        # save the grapher model : every time
        torch.save({'state_dict': grapher_state_dict}, self.grapher_ckpt_path)
        print(f"Saved Grapher checkpoint to {self.grapher_ckpt_path}")

        # save the rgcn model : every time
        torch.save({'state_dict': rgcn_state_dict}, self.rgcn_ckpt_path)
        print(f"Saved RGCN checkpoint to {self.rgcn_ckpt_path}")

def main(args):
    import torch

    args.eval_dir = os.path.join(args.default_root_dir, args.dataset + '_version_' + args.version)
    args.checkpoint_dir = os.path.join(args.eval_dir, 'checkpoints')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(args.eval_dir, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(args.eval_dir, 'test'), exist_ok=True)

    TB = pl_loggers.TensorBoardLogger(save_dir=args.default_root_dir, name='', version=args.dataset + '_version_' + args.version, default_hp_metric=False)


    if args.checkpoint_grapher_model_id < 0:
        grapher_ckpt_path = os.path.join(args.checkpoint_dir, 'grapher_last.ckpt')
    else:
        grapher_ckpt_path = os.path.join(args.checkpoint_dir, f"grapher_epoch={args.checkpoint_grapher_model_id}.ckpt")

    if args.checkpoint_rgcn_model_id < 0:
        rgcn_ckpt_path = os.path.join(args.checkpoint_dir, 'rgcn_last.ckpt')
    else:
        rgcn_ckpt_path = os.path.join(args.checkpoint_dir, f"rgcn_epoch={args.checkpoint_rgcn_model_id}.ckpt")

    if args.freeze_pregrapher:
        print("Freezing Pre-Grapher model...")
        grapher_ckpt_path += grapher_ckpt_path.replace('.ckpt', '_f.ckpt')
        rgcn_ckpt_path += rgcn_ckpt_path.replace('.ckpt', '_f.ckpt')

    # we are always using edges_as_classes
    assert args.edges_as_classes

    if args.run == 'train':
        print("TRAIN MODE")

        dm = GraphDataModule(tokenizer_class=T5Tokenizer,
                             tokenizer_name=args.pretrained_model,
                             cache_dir=args.cache_dir,
                             data_path=args.data_path,
                             dataset=args.dataset,
                             batch_size=args.batch_size,
                             num_data_workers=args.num_data_workers,
                             max_nodes=args.max_nodes,
                             max_edges=args.max_edges,
                             edges_as_classes=args.edges_as_classes,
                             model_max_length=args.model_max_length)

        dm.prepare_data()
        dm.setup(stage='fit')
        dm.setup(stage='validate')

        checkpoint_callback = SeparateCheckpoint(
            grapher_ckpt_path=grapher_ckpt_path,
            rgcn_ckpt_path=rgcn_ckpt_path,
            dirpath=args.checkpoint_dir,
            filename='model-{step}',
            save_last=True,
            save_top_k=-1,
        )

        grapher = LitGrapher(transformer_class=T5ForConditionalGeneration,
                             transformer_name=args.pretrained_model,
                             tokenizer=dm.tokenizer,
                             cache_dir=args.cache_dir,
                             max_nodes=args.max_nodes,
                             max_edges=args.max_edges,
                             edges_as_classes=args.edges_as_classes,
                             default_seq_len_edge=args.default_seq_len_edge,
                             num_classes=len(dm.dataset_train.edge_classes),
                             dropout_rate=args.dropout_rate,
                             num_layers=args.num_layers,
                             vocab_size=len(dm.tokenizer.get_vocab()),
                             bos_token_id=dm.tokenizer.pad_token_id,
                             eos_token_id=dm.tokenizer.eos_token_id,
                             nonode_id=dm.tokenizer.convert_tokens_to_ids('__no_node__'),
                             noedge_id=dm.tokenizer.convert_tokens_to_ids('__no_edge__'),
                             node_sep_id=dm.tokenizer.convert_tokens_to_ids('__node_sep__'),
                             noedge_cl=len(dm.dataset_train.edge_classes) - 1,
                             edge_classes=dm.dataset_train.edge_classes,
                             focal_loss_gamma=args.focal_loss_gamma,
                             eval_dir=args.eval_dir,
                             lr=args.lr,
                             add_rgcn=args.add_rgcn,
                             rgcn_hidden_dim=args.rgcn_hidden_dim,
                             rgcn_layers_num=args.rgcn_layers_num)
        
        # loading the checkpoints
        if os.path.exists(grapher_ckpt_path):
            print(f"Loading Grapher checkpoint from {grapher_ckpt_path}")
            grapher.model.grapher.load_state_dict(torch.load(grapher_ckpt_path)['state_dict'])
        else:
            print(f"No Grapher checkpoint found at {grapher_ckpt_path}, starting from scratch")

        if os.path.exists(rgcn_ckpt_path):
            print(f"Loading RGCN checkpoint from {rgcn_ckpt_path}")
            grapher.model.rgcn.load_state_dict(torch.load(rgcn_ckpt_path)['state_dict'])
        else:
            print(f"No RGCN checkpoint found at {rgcn_ckpt_path}, starting from scratch")

        trainer = pl.Trainer.from_argparse_args(args,
                                                logger=TB,
                                                callbacks=[checkpoint_callback, RichProgressBar(10)])

        dm.setup(stage='validate')

        trainer.fit(model=grapher, datamodule=dm, ckpt_path=None)

    elif args.run == 'test':
        print("TEST MODE")

        if not (os.path.exists(grapher_ckpt_path) and os.path.exists(rgcn_ckpt_path)):
            raise FileNotFoundError(f"Missing Grapher or RGCN checkpoint:\n{grapher_ckpt_path}\n{rgcn_ckpt_path}")

        dm = GraphDataModule(tokenizer_class=T5Tokenizer,
                             tokenizer_name=args.pretrained_model,
                             cache_dir=args.cache_dir,
                             data_path=args.data_path,
                             dataset=args.dataset,
                             batch_size=args.batch_size,
                             num_data_workers=args.num_data_workers,
                             max_nodes=args.max_nodes,
                             max_edges=args.max_edges,
                             edges_as_classes=args.edges_as_classes,
                             model_max_length=args.model_max_length)
        dm.setup(stage='test')

        grapher = LitGrapher(
            transformer_class=T5ForConditionalGeneration,
            transformer_name=args.pretrained_model,
            tokenizer=dm.tokenizer,
            cache_dir=args.cache_dir,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            edges_as_classes=args.edges_as_classes,
            default_seq_len_edge=args.default_seq_len_edge,
            num_classes=len(dm.dataset_test.edge_classes),
            dropout_rate=args.dropout_rate,
            num_layers=args.num_layers,
            vocab_size=len(dm.tokenizer.get_vocab()),
            bos_token_id=dm.tokenizer.pad_token_id,
            eos_token_id=dm.tokenizer.eos_token_id,
            nonode_id=dm.tokenizer.convert_tokens_to_ids('__no_node__'),
            noedge_id=dm.tokenizer.convert_tokens_to_ids('__no_edge__'),
            node_sep_id=dm.tokenizer.convert_tokens_to_ids('__node_sep__'),
            noedge_cl=len(dm.dataset_test.edge_classes) - 1,
            edge_classes=dm.dataset_test.edge_classes,
            focal_loss_gamma=args.focal_loss_gamma,
            eval_dir=args.eval_dir,
            lr=args.lr,
            add_rgcn=args.add_rgcn,
            rgcn_hidden_dim=args.rgcn_hidden_dim,
            rgcn_layers_num=args.rgcn_layers_num,
        )

        grapher.model.grapher.load_state_dict(torch.load(grapher_ckpt_path)['state_dict'])
        grapher.model.rgcn.load_state_dict(torch.load(rgcn_ckpt_path)['state_dict'])

        trainer = pl.Trainer.from_argparse_args(args, logger=TB)

        trainer.test(grapher, datamodule=dm)

    else: # single inference

        if not (os.path.exists(grapher_ckpt_path) and os.path.exists(rgcn_ckpt_path)):
            raise FileNotFoundError(f"Missing Grapher or RGCN checkpoint:\n{grapher_ckpt_path}\n{rgcn_ckpt_path}")

        dm = GraphDataModule(tokenizer_class=T5Tokenizer,
                             tokenizer_name=args.pretrained_model,
                             cache_dir=args.cache_dir,
                             data_path=args.data_path,
                             dataset=args.dataset,
                             batch_size=args.batch_size,
                             num_data_workers=args.num_data_workers,
                             max_nodes=args.max_nodes,
                             max_edges=args.max_edges,
                             edges_as_classes=args.edges_as_classes,
                             model_max_length=args.model_max_length)
        dm.setup(stage='predict')

        grapher = LitGrapher(
            transformer_class=T5ForConditionalGeneration,
            transformer_name=args.pretrained_model,
            tokenizer=dm.tokenizer,
            cache_dir=args.cache_dir,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            edges_as_classes=args.edges_as_classes,
            default_seq_len_edge=args.default_seq_len_edge,
            num_classes=len(dm.dataset_inference.edge_classes),
            dropout_rate=args.dropout_rate,
            num_layers=args.num_layers,
            vocab_size=len(dm.tokenizer.get_vocab()),
            bos_token_id=dm.tokenizer.pad_token_id,
            eos_token_id=dm.tokenizer.eos_token_id,
            nonode_id=dm.tokenizer.convert_tokens_to_ids('__no_node__'),
            noedge_id=dm.tokenizer.convert_tokens_to_ids('__no_edge__'),
            node_sep_id=dm.tokenizer.convert_tokens_to_ids('__node_sep__'),
            noedge_cl=len(dm.dataset_inference.edge_classes) - 1,
            edge_classes=dm.dataset_inference.edge_classes,
            focal_loss_gamma=args.focal_loss_gamma,
            eval_dir=args.eval_dir,
            lr=args.lr,
            add_rgcn=args.add_rgcn,
            rgcn_hidden_dim=args.rgcn_hidden_dim,
            rgcn_layers_num=args.rgcn_layers_num,
        )
        grapher.model.grapher.load_state_dict(torch.load(grapher_ckpt_path)['state_dict'])
        grapher.model.rgcn.load_state_dict(torch.load(rgcn_ckpt_path)['state_dict'])

        tokenizer = T5Tokenizer.from_pretrained(grapher.transformer_name, cache_dir=grapher.cache_dir, model_max_length=args.model_max_length)
        tokenizer.add_tokens('__no_node__')
        tokenizer.add_tokens('__no_edge__')
        tokenizer.add_tokens('__node_sep__')

        text_tok = tokenizer([args.inference_input_text],
                             add_special_tokens=True,
                             padding=True,
                             return_tensors='pt')

        text_input_ids, mask = text_tok['input_ids'], text_tok['attention_mask']

        _, seq_nodes, _, seq_edges = grapher.model.sample(text_input_ids, mask)

        dec_graph = decode_graph(tokenizer, grapher.edge_classes, seq_nodes, seq_edges, grapher.edges_as_classes,
                                grapher.node_sep_id, grapher.max_nodes, grapher.noedge_cl, grapher.noedge_id,
                                grapher.bos_token_id, grapher.eos_token_id)

        graph_str = ['-->'.join(tri) for tri in dec_graph[0]]

        print(f'Generated Graph: {graph_str}')


if __name__ == "__main__":

    # Parsing arguments
    parser = argparse.ArgumentParser(description='Arguments')

    parser.add_argument("--dataset", type=str, default='webnlg')
    parser.add_argument("--run", type=str, default='train')
    parser.add_argument('--pretrained_model', type=str, default='t5-small')
    parser.add_argument('--version', type=str, default='0')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--num_data_workers', type=int, default=3)
    parser.add_argument('--checkpoint_grapher_model_id', type=int, default=-1)
    parser.add_argument('--checkpoint_rgcn_model_id', type=int, default=-1)
    parser.add_argument('--max_nodes', type=int, default=8)
    parser.add_argument('--max_edges', type=int, default=7)
    parser.add_argument('--default_seq_len_node', type=int, default=20)
    parser.add_argument('--default_seq_len_edge', type=int, default=20)
    parser.add_argument('--edges_as_classes', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument("--focal_loss_gamma", type=float, default=0.0)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--eval_dump_only", type=int, default=0)
    parser.add_argument("--inference_input_text", type=str,
                        default='Danielle Harris had a main role in Super Capers, a 98 minute long movie.')
    parser.add_argument("--model_max_length", type=int,
                        default=512)
    parser.add_argument("--rgcn_layers_num", type=int,
                        default=2)
    parser.add_argument("--checkpoint_step_frequency", type=int, default=1000)
    parser.add_argument("--rgcn_hidden_dim", type=int, default=128)
    parser.add_argument("--add-rgcn", action='store_true')


    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    pprint.pp(f"arguments: \n{args}")


    main(args)
