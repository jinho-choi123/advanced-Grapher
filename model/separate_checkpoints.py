import torch
import os
import subprocess
from pathlib import Path
import zipfile
import io
from model.litgrapher import LitGrapher

def get_repo_root():
    """
    Return the Git repository root directory. Assuming that this file is in the `model/` directory.
    """
    try:
        git_root = subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return git_root
    except Exception:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve_path(path_str):
    """
    Resolve a checkpoint path relative to the repo's `checkpoints/` directory if not absolute.
    """
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    repo_root = get_repo_root()
    return os.path.join(repo_root, 'checkpoints', path_str)

def load_state_dict(ckpt_path, map_location):
    """
    Load a state dictionary from a checkpoint file.
    """
    try:
        with zipfile.ZipFile(ckpt_path, 'r') as z:
            candidates = [name for name in z.namelist() if 'state_dict' in name]
            if not candidates:
                raise KeyError
            entry = 'state_dict' if 'state_dict' in candidates else candidates[0]
            raw = z.read(entry)
        sd = torch.load(io.BytesIO(raw), map_location=map_location)
        return sd
    except zipfile.BadZipFile:
        ckpt = torch.load(ckpt_path, map_location=map_location)
        return ckpt.get('state_dict', ckpt)
    except KeyError:
        raise KeyError(f"No state_dict found in checkpoint: {ckpt_path}")

def separate_checkpoints(combined_ckpt_path, grapher_ckpt_path, rgcn_ckpt_path, device=0):
    """
    Load a combined checkpoint and separate it into Grapher and R-GCN checkpoints.
    The reference directory is set to the checkpoints/
    """
    combined_path = resolve_path(combined_ckpt_path)
    print(f"Combined checkpoint path: {combined_path}")
    grapher_path = resolve_path(grapher_ckpt_path)
    rgcn_path = resolve_path(rgcn_ckpt_path)

    map_location = f'cuda:{device}' if device >= 0 else 'cpu'

    combined_model: LitGrapher = LitGrapher.load_from_checkpoint(combined_path, map_location=map_location, strict=False)

    grapher = combined_model.model.grapher if hasattr(combined_model, 'model') else combined_model.grapher
    rgcn = combined_model.model.rgcn if hasattr(combined_model, 'model') else combined_model.rgcn
    
    torch.save({'state_dict': grapher.state_dict()}, str(grapher_path))
    torch.save({'state_dict': rgcn.state_dict()}, str(rgcn_path))

    print(f"Saved Grapher checkpoint: {grapher_path}")
    print(f"Saved R-GCN checkpoint: {rgcn_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Separate combined Lightning checkpoint into Grapher and R-GCN components.')
    parser.add_argument('--combined_ckpt', type=str, required=True,
                        help='Path to the combined checkpoint file')
    parser.add_argument('--grapher_ckpt', type=str, required=True,
                        help='Output path for the Grapher checkpoint')
    parser.add_argument('--rgcn_ckpt', type=str, required=True,
                        help='Output path for the R-GCN checkpoint')
    parser.add_argument('--device', type=int, default=0,
                        help='Device ID for loading the checkpoint (default: 0, -1 for CPU)')

    args = parser.parse_args()
    separate_checkpoints(args.combined_ckpt, args.grapher_ckpt, args.rgcn_ckpt, args.device)