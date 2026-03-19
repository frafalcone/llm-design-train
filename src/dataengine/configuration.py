context_size = 1024
batch = 8
num_workers = 4
percentage = 1.0
trn_bin = "data/bin/train.bin"
val_bin = "data/bin/validation.bin"
trn_parquet = "data/parq/train.parquet"
val_parquet = "data/parq/validation.parquet"

data_configuration = {
    "context_size": context_size,
    "batch": batch,
    "num_workers": num_workers,
    "percentage" : percentage,
    "trn_bin": trn_bin,
    "val_bin": val_bin,
    "trn_parquet": trn_parquet,
    "val_parquet": val_parquet,
}
