import os
import torch
import tkinter as tk
from tkinter import filedialog
from model import RubikDistancePredictor

def shrink_checkpoint():
    # Hide the main root Tk window
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    # Prompt user to select a checkpoint
    ckpt_path = filedialog.askopenfilename(
        title="Select Lightning Checkpoint to Shrink",
        filetypes=[("PyTorch Checkpoint", "*.ckpt *.pt"), ("All Files", "*.*")]
    )

    if not ckpt_path:
        print("No file selected. Exiting.")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    orig_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

    # Load Lightning module state
    model = RubikDistancePredictor.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        strict=False
    )
    model.eval()

    # Extract only network state dict and rotation perms buffer
    shrunk_payload = {
        "state_dict": model.network.state_dict(),
        "rotation_perms": model.rotation_perms.cpu(),
        "num_classes": model.hparams.num_classes,
        "hidden_dim": model.hparams.hidden_dim,
    }

    # Construct output path in the same directory
    dir_name, file_name = os.path.split(ckpt_path)
    base_name, _ = os.path.splitext(file_name)
    output_path = os.path.join(dir_name, f"{base_name}_shrunk.pt")

    # Save lightweight weight dictionary
    torch.save(shrunk_payload, output_path)
    
    new_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\nSuccessfully shrunk model!")
    print(f"Original size: {orig_size_mb:.2f} MB")
    print(f"Shrunk size:   {new_size_mb:.2f} MB")
    print(f"Saved to:      {output_path}")

if __name__ == "__main__":
    shrink_checkpoint()