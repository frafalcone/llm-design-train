epochs = 3
learning_rate = 1e-3
weight_decay = 0.2
betas = (0.9, 0.95)
lr_min_ratio = 0.1
load_best = False
load_checkpoint = False
accumulation_steps = 32
vocabulary = 50257
use_scheduler = True
warmup_ratio = 0.1

trn_configuration = {
    "epoch": epochs,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "betas": betas,
    "lr_min_ratio": lr_min_ratio,
    "load_best": load_best,
    "load_checkpoint": load_checkpoint,
    "accumulation_steps": accumulation_steps,
    "vocabulary": vocabulary,
    "use_scheduler": use_scheduler,
    "warmup_ratio": warmup_ratio
}