import torch
from safetensors.torch import save_file

def convert_pth_to_safetensors(pth_file, safetensors_file):
    # Load checkpoint
    checkpoint = torch.load(pth_file, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    save_file(state_dict, safetensors_file)
    print(f"Converted {pth_file} to {safetensors_file}")

Example usage (replace with your own file paths)
pth_file = "path/to/your/model.pth"
safetensors_file = "path/to/your/model.safetensors"
convert_pth_to_safetensors(pth_file, safetensors_file)