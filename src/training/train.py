import torch
import math
from tqdm import tqdm
from utils.state_manager import save_state, load_state
import json

def train_model(model, trn_configuration, training_loader, validation_loader, device):
    model.to(device)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    best_val_loss = float('inf')
    total_target_epochs = trn_configuration.get("epoch", 0)
    learning_rate = trn_configuration.get("learning_rate", 0)
    weight_decay = trn_configuration.get("weight_decay", 0)
    accumulation_steps = trn_configuration.get("accumulation_steps", 1)
    use_scheduler = trn_configuration.get("use_scheduler", False)
    warmup_ratio = trn_configuration.get("warmup_ratio", 0.1)
    adamw_betas = trn_configuration.get("betas", (0.9, 0.95))

    decay_params    = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() <  2 and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ], lr=learning_rate, betas=adamw_betas)
    
    use_amp = device.type in ('cuda')
    use_bf16 = False
    if device.type == 'cuda':
        try:
            _t = torch.zeros(1, dtype=torch.bfloat16, device=device)
            _ = _t + _t
            use_bf16 = True
            del _t, _
        except Exception:
            use_bf16 = False
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if device.type != 'cuda':
        amp_dtype = torch.float32

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and not use_bf16))

    total_batches = len(training_loader)
    steps_per_epoch = math.ceil(total_batches / accumulation_steps)
    total_optimization_steps = steps_per_epoch * total_target_epochs
    warmup_steps = max(0, int(total_optimization_steps * warmup_ratio))
    
    lr_min_ratio = trn_configuration.get("lr_min_ratio", 0.1)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_optimization_steps - warmup_steps))
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(lr_min_ratio, cosine_val)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda if use_scheduler else (lambda step: 1.0))
    
    filepath_checkpoint, filepath_best = "output/lckpt.pth", "output/bckpt.pth"
    start_epoch, start_batch, current_opt_step, last_loss = 0, -1, 0, 0.0

    target_path = None
    if trn_configuration.get("load_best"): target_path = filepath_best
    elif trn_configuration.get("load_checkpoint"): target_path = filepath_checkpoint

    if target_path:
        start_epoch, start_batch, current_opt_step, last_loss, best_val_loss = load_state(
            model, optimizer, scheduler, scaler, target_path, device
        )

    if current_opt_step >= total_optimization_steps: return

    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(start_epoch, total_target_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_steps = 0
        save_step = max(1, total_batches // 5)

        skip_batches = (start_batch + 1) if epoch == start_epoch and start_batch > -1 else 0
        
        with tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{total_target_epochs}", unit="it") as pbar:
            if skip_batches:
                pbar.update(skip_batches)

            loader_iter = iter(training_loader)
            for _ in range(skip_batches):
                next(loader_iter, None)

            for batch_idx, (inputs, targets) in enumerate(loader_iter, start=skip_batches):
                inputs, targets = inputs.to(device), targets.to(device)

                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(inputs)
                    loss = criterion(logits.flatten(0, 1), targets.flatten())
                    distorted_loss = loss / accumulation_steps
                
                scaler.scale(distorted_loss).backward()
                running_loss += loss.item()
                running_steps += 1
                
                is_last_batch = (batch_idx + 1 == total_batches)
                is_update_step = ((batch_idx + 1) % accumulation_steps == 0) or is_last_batch

                if is_update_step:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        current_opt_step += 1

                        last_loss = running_loss / running_steps
                        
                        pbar.set_postfix({
                            "loss": f"{last_loss:.4f}", 
                            "gnorm": f"{grad_norm:.3f}",
                            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                            "best_v": f"{best_val_loss:.4f}" if best_val_loss != float('inf') else "N/A"
                        })

                        running_loss = 0.0
                        running_steps = 0
                        optimizer.zero_grad(set_to_none=True)

                if (batch_idx + 1) % save_step == 0 or is_last_batch:
                    save_state(model, optimizer, scheduler, scaler, epoch, batch_idx, 
                                current_opt_step, last_loss, best_val_loss, filepath_checkpoint)

                pbar.update(1)
        
        v_loss = validate_model(model, validation_loader, criterion, device, use_amp, amp_dtype)
        model.train()

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_state(model, optimizer, scheduler, scaler, epoch, len(training_loader)-1, 
                        current_opt_step, last_loss, best_val_loss, filepath_best, True)
            
    training_results = {
        "best_validation_loss": best_val_loss,
        "final_training_loss": last_loss,
        "total_optimization_steps": current_opt_step,
        "device": str(device)
    }

    results_path = "output/training_results.json"
    with open(results_path, "w") as f:
        json.dump(training_results, f, indent=4)



def validate_model(model, val_loader, criterion, device, use_amp=False, amp_dtype=torch.float16):
    model.eval()
    total_loss, total_tokens = 0.0, 0

    with tqdm(total=len(val_loader), desc="Validating", unit="it", leave=False) as val_pbar:
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            for inputs, targets in val_loader: 
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits.flatten(0, 1), targets.flatten())

                if torch.isnan(loss) or torch.isinf(loss):
                    return float('inf')

                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()
                val_pbar.update(1)

    return total_loss / total_tokens if total_tokens > 0 else float('inf')