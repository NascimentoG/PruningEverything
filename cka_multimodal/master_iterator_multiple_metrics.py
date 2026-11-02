import argparse
import os
import random
import torch
import gc
import numpy as np
import csv
from transformers import LlavaForConditionalGeneration, AutoProcessor

from prune_groups_multiple_metrics import prune_groups
from prune_layers_multiple_metrics import prune_layers
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
    
    if args.metric in ["kl", "cosine", "cka"]:
        print(f"Loading tuned original model for {args.metric} metric reference...")
        reference_model = LlavaForConditionalGeneration.from_pretrained(
            args.tuned_original_model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        reference_processor = AutoProcessor.from_pretrained(args.tuned_original_tokenizer)
        
        if args.metric == "cka":
            reference_cka = compute_model_representation(reference_model, reference_processor, args.device)

        del reference_model, reference_processor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        reference_model = None
        reference_processor = None
        gc.collect()
        print(f"Reference data for {args.metric} metric extracted successfully")

    for iteration in range(1, args.num_iterations + 1):
        print('#'*100)
        print('#'*45, iteration, '#'*45)

        if iteration > 1:
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

        print("Clearing GPU memory after iteration...")
        gc.collect()
        torch.cuda.empty_cache()

        print('#'*100)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    parser.add_argument('--model_name', type=str, default='llava-hf/llava-1.5-7b-hf', help='base model name')
    parser.add_argument('--save_dir', type=str, required=True, 
                        help='Save directory')

    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--num_iterations', type=int, default=20, help='number of iterations')
    

    parser.add_argument('--tuned_original_model', type=str, default="llava-hf/llava-1.5-7b-hf",
                        help='Path to the fine-tuned original model for reference')
    parser.add_argument('--tuned_original_tokenizer', type=str, default="llava-hf/llava-1.5-7b-hf",
                        help='Path to the fine-tuned original tokenizer for reference')
    parser.add_argument('--metric', type=str, default="cka", choices=["ppl", "kl", "cosine", "cka"]),

    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
