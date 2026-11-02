import gc

from cka import compute_model_representation, compute_cka_from_reprs
from transformers import LlavaForConditionalGeneration
from pathlib import Path
import torch
import sys
import subprocess
import shutil

def tune_model(model, tokenizer, training_steps, eval_metric, reference_probs, reference_logits, reference_representation):
    temp_prune_model_dir = Path(str(Path(__file__).resolve().parent /"temp_prune_model"))
    temp_prune_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(temp_prune_model_dir)
    tokenizer.save_pretrained(temp_prune_model_dir)
    model.to("cpu")
    del model

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

    model = None


    temp_tuned_model_dir = Path(str(Path(__file__).resolve().parent /"temp_tuned_model"))   
    temp_tuned_model_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(str(Path(__file__).resolve().parent / "llava_alpaca_llm_ft.py"))
    print(f"CUDA memory allocated before partial ft: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

    cmd = [
        sys.executable,
        str(script_path),
        "--llava_ckpt", str(temp_prune_model_dir),
        "--output_llava", str(temp_tuned_model_dir),
        "--work_dir","/mnt/sdb/keith/mop_cka_multimodal/dir" ,
        "--max_steps", str(training_steps),
        "--epochs", str(1),
        "--save_steps", str(1000),
        "--logging_steps", str(10),
    ]

    print("Running fine-tune:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"CUDA memory allocated after partial ft: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

    model = LlavaForConditionalGeneration.from_pretrained(temp_tuned_model_dir, torch_dtype=torch.bfloat16, device_map="cuda")
    if eval_metric == "cka":
        pruned_representation = compute_model_representation(model, tokenizer, model.device)
        eval = compute_cka_from_reprs(reference_representation, pruned_representation)
    else:
        raise ValueError(f"Unsupported evaluation metric: {eval_metric}")

    if temp_prune_model_dir.exists():
                shutil.rmtree(temp_prune_model_dir, ignore_errors=True)
    if temp_tuned_model_dir.exists():
                shutil.rmtree(temp_tuned_model_dir, ignore_errors=True)

    return eval