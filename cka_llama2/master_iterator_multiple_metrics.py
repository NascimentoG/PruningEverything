import argparse
import os
import random
import torch
import gc
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from prune_groups_amp import prune_groups
from prune_layers_amp import prune_layers
from cka import compute_model_representation

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    set_random_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Seed: {args.seed}")
    
    reference_probs = None
    reference_logits = None
    reference_cka = None
    
    if args.metric in ["cka"]:
        print(f"Loading tuned original model for {args.metric} metric reference...")
        reference_model = AutoModelForCausalLM.from_pretrained(
            args.tuned_original_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        reference_tokenizer = AutoTokenizer.from_pretrained(args.tuned_original_tokenizer)
        
        if args.metric == "cka":
            reference_cka = compute_model_representation(reference_model, reference_tokenizer, args.device)

        del reference_model, reference_tokenizer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        reference_model = None
        reference_tokenizer = None
        gc.collect()
        print(f"Reference data for {args.metric} metric extracted successfully")
    
    for iteration in range(1, args.num_iterations + 1):
        if iteration <= args.start_iteration:
            continue
        print('#'*100)
        print('#'*45, iteration, '#'*45)

        if iteration == 1:
            args.model_name = args.model_name
        else:
            args.model_name = args.save_dir + '/' + f'prune_iteration_{iteration-1}'
        
        metric_layers = prune_layers(args, iteration, args.save_dir, eval_metric=args.metric, 
                                     tune_eval=True, reference_probs=reference_probs, reference_logits=reference_logits, reference_representation=reference_cka)
        print("Metric Layers:", metric_layers)
        metric_groups = prune_groups(args, iteration, args.save_dir, eval_metric=args.metric, 
                                     tune_eval=True, reference_probs=reference_probs, reference_logits=reference_logits, reference_representation=reference_cka)
        print("Metric Neurons:", metric_groups)
        if metric_layers < metric_groups:
            prune_layers(args, iteration, args.save_dir, eval_metric=args.metric, 
                         tune_eval=False, reference_probs=reference_probs, reference_logits=reference_logits, reference_representation=reference_cka)
            print(f'ITERATION {iteration} PRUNED LAYER')
        else:
            prune_groups(args, iteration, args.save_dir, eval_metric=args.metric, 
                         tune_eval=False, reference_probs=reference_probs, reference_logits=reference_logits, reference_representation=reference_cka)
            print(f'ITERATION {iteration} PRUNED GROUP')

        print('#'*100)


if __name__ == "__main__":
    os.environ["HUGGINGFACE_TOKEN"] = ""
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf', help='base model name')
    parser.add_argument('--save_dir', type=str, default="./save_dir", help='Save directory')
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--num_iterations', type=int, default=20, help='number of iterations')
    parser.add_argument('--metric', type=str, default="cka", choices=["cka"], help='Evaluation metric to use for pruning decisions')
    parser.add_argument('--tuned_original_model', type=str, default="./alpaca_ft_original", help='Path to the fine-tuned original model')
    parser.add_argument('--tuned_original_tokenizer', type=str, default="./alpaca_ft_original", help='Path to the fine-tuned original tokenizer')
    parser.add_argument('--start_iteration', type=int, default=0, help='Iteration number to start from (useful for resuming)')
    
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
