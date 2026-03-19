import pandas as pd
import glob
from tqdm import tqdm
import os
import numpy

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def process_parquet_data(input_path, output_bin, tokenizer, batch_size=2000):
    if os.path.isdir(input_path):
        parquet_files = glob.glob(os.path.join(input_path, "*.parquet"))
    else:
        parquet_files = [input_path]

    print(f"Found {len(parquet_files)} file(s). Target: {output_bin}")

    eos_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    with open(output_bin, 'wb') as f_bin:
        for file_path in parquet_files:
            print(f"\nProcessing: {os.path.basename(file_path)}")
            df = pd.read_parquet(file_path, engine='pyarrow', columns=['text'])

            for i in tqdm(range(0, len(df), batch_size), desc="Tokenizing"):
                batch_text = df['text'].iloc[i : i + batch_size].tolist()

                encoded_batch = tokenizer.encode_ordinary_batch(batch_text)

                all_ids = [token for ids in encoded_batch for token in (ids + [eos_id])]

                if all_ids:
                    f_bin.write(numpy.array(all_ids, dtype=numpy.int32).tobytes())

    print(f"\nSuccess! Saved to: {output_bin}")


def pre_tokenize_data(data_configuration, tokenizer):
    trn_bin = data_configuration.get("trn_bin", "")
    val_bin = data_configuration.get("val_bin", "")
    trn_input = data_configuration.get("trn_parquet", "")
    val_input = data_configuration.get("val_parquet", "")

    for input_p, output_b, label in [(trn_input, trn_bin, "Training"), (val_input, val_bin, "Validation")]:
        if input_p and os.path.exists(input_p):
            print(f"--- Pre-processing {label} Set ---")
            process_parquet_data(input_p, output_b, tokenizer)

    return trn_bin, val_bin