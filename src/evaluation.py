import inspect
import random
import traceback
from collections import defaultdict

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from architecture.model import Model
from architecture.configuration import model_configuration
from utils.model_manager import load_model

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def safe_load(name, *args, **kwargs):
    try:
        return load_dataset(name, *args, **kwargs)
    except Exception:
        return load_dataset(name, *args, trust_remote_code=True, **kwargs)


def to_list(batch):
    return [dict(zip(batch, t)) for t in zip(*batch.values())]


def make_result(scores_per_cat: dict) -> dict:
    result = dict(scores_per_cat)
    result["_overall"] = float(np.mean(list(scores_per_cat.values()))) if scores_per_cat else 0.0
    return result


class CustomWrapper:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.is_tiktoken = True

    def encode(self, text: str) -> list:
        return self.tokenizer.encode(text)

    def decode_single(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])


class GPT2Wrapper:
    def __init__(self, device):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.model.eval()
        self.device = device
        self.is_tiktoken = False

    def encode(self, text: str) -> list:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_single(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])


def _get_pad_id(model) -> int:
    if model.is_tiktoken:
        return 0
    pid = getattr(model.tokenizer, "pad_token_id", None)
    return pid if pid is not None else 50256


def batch_logprob(model, contexts: list, continuations: list = None) -> np.ndarray:
    pad_id = _get_pad_id(model)
    all_ids, all_masks = [], []

    for i, ctx in enumerate(contexts):
        ctx_tok = model.encode(ctx)
        if continuations is not None:
            cont_tok = model.encode(continuations[i])
            ids  = ctx_tok + cont_tok
            mask = [0] * len(ctx_tok) + [1] * len(cont_tok)
        else:
            ids  = ctx_tok
            mask = [1] * len(ctx_tok)
        all_ids.append(ids)
        all_masks.append(mask)

    max_len = max(len(t) for t in all_ids)
    pad_ids, pad_masks = [], []
    for ids, m in zip(all_ids, all_masks):
        p = max_len - len(ids)
        pad_ids.append([pad_id] * p + ids)
        pad_masks.append([0] * p + m)

    input_ids  = torch.tensor(pad_ids,   dtype=torch.long).to(model.device)
    attn_mask  = (input_ids != pad_id).long()
    score_mask = torch.tensor(pad_masks, dtype=torch.long).to(model.device)

    kwargs = {}
    if "attention_mask" in inspect.signature(model.model.forward).parameters:
        kwargs["attention_mask"] = attn_mask

    with torch.no_grad():
        out    = model.model(input_ids, **kwargs)
        logits = out.logits if hasattr(out, "logits") else out

    sl = logits[:, :-1, :].contiguous()
    tl = input_ids[:, 1:].contiguous()
    sm = score_mask[:, 1:].contiguous()

    lp       = F.log_softmax(sl, dim=-1)
    gathered = torch.gather(lp, 2, tl.unsqueeze(-1)).squeeze(-1) * sm
    scores   = gathered.sum(dim=1)
    lengths  = sm.sum(dim=1).clamp(min=1)
    return (scores / lengths).cpu().numpy()


def batch_score_mc(model, contexts: list, choices_list: list) -> list:
    preds = []
    for ctx, choices in zip(contexts, choices_list):
        scores = batch_logprob(model, [ctx] * len(choices), choices)
        preds.append(int(np.argmax(scores)))
    return preds


def evaluate_blimp(model, batch_size: int = 32) -> dict:
    configs = get_dataset_config_names("nyu-mll/blimp")
    scores = {}

    for config in tqdm(configs, desc="BLiMP", leave=False):
        dataset = safe_load("nyu-mll/blimp", config, split="train")
        correct = 0
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i: i + batch_size]
            g_sc  = batch_logprob(model, batch["sentence_good"])
            b_sc  = batch_logprob(model, batch["sentence_bad"])
            correct += int(np.sum(g_sc > b_sc))
        scores[config] = correct / len(dataset) * 100

    torch.cuda.empty_cache()
    return make_result(scores)


def evaluate_cola(model, batch_size: int = 32) -> dict:
    scores = {}
    splits_to_try = [
        ("in_domain",     "validation"),
        ("out_of_domain", "test"),
    ]

    for cat_name, split_name in splits_to_try:
        try:
            dataset = safe_load("nyu-mll/glue", "cola", split=split_name)
        except Exception:
            continue

        sentences = dataset["sentence"]
        labels    = dataset["label"]
        all_sc    = []
        for i in tqdm(range(0, len(sentences), batch_size),
                      desc=f"CoLA-{cat_name}", leave=False):
            all_sc.extend(batch_logprob(model, sentences[i: i + batch_size]).tolist())

        threshold        = float(np.median(all_sc))
        preds            = [1 if s > threshold else 0 for s in all_sc]
        scores[cat_name] = sum(p == l for p, l in zip(preds, labels)) / len(labels) * 100

    torch.cuda.empty_cache()
    return make_result(scores)


def evaluate_lambada(model, batch_size: int = 16) -> dict:
    dataset = safe_load("lambada", split="validation")
    correct = 0
    total   = 0

    for i in tqdm(range(0, len(dataset), batch_size), desc="LAMBADA", leave=False):
        batch = dataset[i: i + batch_size]
        for text in batch["text"]:
            text = text.rstrip()
            idx  = text.rfind(" ")
            if idx == -1:
                continue
            prefix = text[:idx]
            target = text[idx:]

            ids       = model.encode(prefix)
            input_ids = torch.tensor([ids], dtype=torch.long).to(model.device)
            kwargs    = {}
            if "attention_mask" in inspect.signature(model.model.forward).parameters:
                kwargs["attention_mask"] = torch.ones_like(input_ids)

            with torch.no_grad():
                out    = model.model(input_ids, **kwargs)
                logits = out.logits if hasattr(out, "logits") else out

            pred_id   = int(torch.argmax(logits[0, -1, :]).item())
            pred_word = model.decode_single(pred_id)
            if pred_word.strip().lower() == target.strip().lower():
                correct += 1
            total += 1

    torch.cuda.empty_cache()
    return make_result({"last_word_accuracy": correct / total * 100 if total > 0 else 0.0})


def evaluate_hellaswag(model, batch_size: int = 8) -> dict:
    dataset     = safe_load("hellaswag", split="validation")
    cat_correct = defaultdict(int)
    cat_total   = defaultdict(int)

    for i in tqdm(range(0, len(dataset), batch_size), desc="HellaSwag", leave=False):
        batch = to_list(dataset[i: i + batch_size])
        preds = batch_score_mc(
            model,
            [x["ctx"] for x in batch],
            [x["endings"] for x in batch],
        )
        for p, x in zip(preds, batch):
            cat = x.get("activity_label") or "unknown"
            cat_correct[cat] += int(p == int(x["label"]))
            cat_total[cat]   += 1

    scores = {cat: cat_correct[cat] / cat_total[cat] * 100 for cat in cat_total}
    torch.cuda.empty_cache()
    return make_result(scores)


def evaluate_arc(model, difficulty: str, batch_size: int = 8) -> dict:
    dataset     = safe_load("ai2_arc", difficulty, split="validation")
    cat_correct = defaultdict(int)
    cat_total   = defaultdict(int)

    for i in tqdm(range(0, len(dataset), batch_size),
                  desc=f"ARC-{difficulty}", leave=False):
        batch  = to_list(dataset[i: i + batch_size])
        labels = []
        for x in batch:
            l_str    = str(x["answerKey"])
            possible = x["choices"]["label"]
            labels.append(possible.index(l_str))

        preds = batch_score_mc(
            model,
            [x["question"] for x in batch],
            [x["choices"]["text"] for x in batch],
        )
        for p, l, x in zip(preds, labels, batch):
            cat = x.get("subject") or difficulty
            cat_correct[cat] += int(p == l)
            cat_total[cat]   += 1

    scores = {cat: cat_correct[cat] / cat_total[cat] * 100 for cat in cat_total}
    torch.cuda.empty_cache()
    return make_result(scores)


def _fmt(v) -> str:
    if v is None:            return "ERR"
    if v == "PENDING":       return "PENDING"
    if isinstance(v, float): return f"{v:.2f}"
    return str(v)


def save_results_incremental(custom_res: dict, gpt2_res: dict):
    all_tasks = list(custom_res.keys())

    with open("benchmark_results.txt", "w", encoding="utf-8") as f:
        f.write("╔══════════════════════════════════════════════════════╗\n")
        f.write("║              BENCHMARK RESULTS                       ║\n")
        f.write("╚══════════════════════════════════════════════════════╝\n\n")

        for model_name, res in [("CUSTOM", custom_res), ("GPT-2", gpt2_res)]:
            f.write(f"{'─'*56}\n")
            f.write(f"  MODEL: {model_name}\n")
            f.write(f"{'─'*56}\n")

            for task in all_tasks:
                val = res.get(task)

                if val == "PENDING":
                    f.write(f"\n  [{task}]  → PENDING\n")

                elif val is None:
                    f.write(f"\n  [{task}]  → ERROR / SKIPPED\n")

                elif isinstance(val, dict):
                    overall = val.get("_overall", 0.0)
                    n_cats  = len(val) - 1
                    f.write(f"\n  [{task}]  overall = {overall:.2f}%  ({n_cats} subcategories)\n")
                    for subcat, score in sorted(val.items()):
                        if subcat == "_overall":
                            continue
                        f.write(f"      {subcat:<50} {score:>7.2f}%\n")
                else:
                    f.write(f"\n  [{task}]  {val:.2f}%\n")

            f.write("\n")




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    try:
        model_custom = Model(model_configuration)
        load_model(model=model_custom, filepath="output/model.pth")
        custom = CustomWrapper(model_custom, tiktoken.get_encoding("gpt2"), device)
        print("Custom model loaded.")
    except Exception as e:
        print(f"Custom load error: {e}")
        return

    gpt2 = GPT2Wrapper(device)
    print("GPT-2 loaded.")

    tasks = {
        "BLiMP":         evaluate_blimp,
        "CoLA":          evaluate_cola,
        "LAMBADA":       evaluate_lambada,
        "HellaSwag":     evaluate_hellaswag,
        "ARC-Easy":      lambda m: evaluate_arc(m, "ARC-Easy"),
        "ARC-Challenge": lambda m: evaluate_arc(m, "ARC-Challenge"),
    }

    res_custom = {k: "PENDING" for k in tasks}
    res_gpt2   = {k: "PENDING" for k in tasks}
    save_results_incremental(res_custom, res_gpt2)

    print("\nStarting benchmarks (incremental saving enabled)...\n")

    for task_name, task_func in tasks.items():
        print(f"{'═'*54}")
        print(f"  {task_name}")
        print(f"{'═'*54}")

        for model_obj, res_dict, label in [
            (custom, res_custom, "Custom"),
            (gpt2,   res_gpt2,   "GPT-2"),
        ]:
            try:
                print(f"   → {label}...")
                result = task_func(model_obj)
                res_dict[task_name] = result
                overall = result.get("_overall", 0.0)
                n_cats  = len(result) - 1
                print(f"      overall = {overall:.2f}%  ({n_cats} subcategories)")
            except Exception:
                print(f"   Error {label} ({task_name}):\n{traceback.format_exc()}")
                res_dict[task_name] = None

        save_results_incremental(res_custom, res_gpt2)
        print("   Files updated.\n")

    W = 60
    print("\n" + "═" * W)
    print(f"  {'FINAL RESULTS (OVERALL)':^{W-4}}")
    print("═" * W)
    print(f"  {'Task':<22} {'Custom':>10} {'GPT-2':>10} {'Diff':>8}")
    print("─" * W)
    for task in tasks:
        c = res_custom.get(task)
        g = res_gpt2.get(task)
        c_ov = c.get("_overall") if isinstance(c, dict) else c
        g_ov = g.get("_overall") if isinstance(g, dict) else g
        c_s  = f"{c_ov:.2f}%" if isinstance(c_ov, float) else str(c_ov)
        g_s  = f"{g_ov:.2f}%" if isinstance(g_ov, float) else str(g_ov)
        d_s  = (f"{c_ov - g_ov:+.2f}"
                if isinstance(c_ov, float) and isinstance(g_ov, float)
                else "N/A")
        print(f"  {task:<22} {c_s:>10} {g_s:>10} {d_s:>8}")
    print("═" * W)
    print("\nDone. See benchmark_results.txt\n")


if __name__ == "__main__":
    main()