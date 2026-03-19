import torch
import os
import gc

def save_model(model, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict()
    }
    tmp_path = filepath + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, filepath)


def load_model(model, filepath):
    if not os.path.exists(filepath):
        return
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    gc.collect()

