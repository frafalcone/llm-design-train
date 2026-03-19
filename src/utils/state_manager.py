import torch
import os
import gc

def save_state(model, optimizer, scheduler, scaler, epoch, batch_idx, current_opt_step, loss, best_val_loss, filepath, best=False):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'epoch': epoch,
        'batch': batch_idx,
        'current_opt_step': current_opt_step,
        'loss': loss,
        'best_val_loss': best_val_loss,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }
    
    tmp_path = filepath + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, filepath)

def load_state(model, optimizer, scheduler, scaler, filepath, device):
    if not os.path.exists(filepath):
        return 0, -1, 0, float('inf'), float('inf')

    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    epoch = checkpoint.get('epoch', 0)
    batch = checkpoint.get('batch', -1)
    current_opt_step = checkpoint.get('current_opt_step', 0)
    loss = checkpoint.get('loss', float('inf'))
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    if 'rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['rng_state'])
    if checkpoint.get('cuda_rng_state') is not None:
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])

    del checkpoint
    gc.collect()

    return epoch, batch, current_opt_step, loss, best_val_loss