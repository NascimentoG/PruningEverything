import os
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from finetuning_multiple_metrics import tune_model

def print_model_parameters(model, message):
    """
    Prints the total number of parameters in the model.

    Args:
        model (nn.Module): The model.
        message (str): A custom message to display.
    """
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{message}: {total_params} parameters")
    return total_params


def prune_model_by_layers(model):
    """
    Prune the last layer of the model while keeping the last two layers intact.
    Args:
        model: The model to prune.
    """
    layer_to_prune = len(model.model.layers) - 3 

    if 0 <= layer_to_prune < len(model.model.layers):
        print(f"Pruning layer {layer_to_prune}...")
        del model.model.layers[layer_to_prune]
    else:
        print(f"Layer index {layer_to_prune} is out of range. Skipping.")

    print(f"Model pruned. Remaining layers: {len(model.model.layers)}")


def prune_layers(args, iteration, base_dir, eval_metric, tune_eval=False, reference_probs=None, reference_logits=None, reference_representation=None):
    torch.cuda.empty_cache()
    
    print("Loading tokenizer from tuned original model...")
    tokenizer = AutoTokenizer.from_pretrained(args.tuned_original_tokenizer)
    
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded from tuned original model.")

    print("Loading model...")
    config = AutoConfig.from_pretrained(args.model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        print(f"Warning during model loading: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True
        )
    print("Model loaded.")

    len_before_pruning = len(model.model.layers)

    prune_model_by_layers(model)

    model.config.num_hidden_layers = len_before_pruning - 1
    model.config.head_dim = 128
    if tune_eval:
        training_steps = 10
        if iteration != 1:
            training_steps = iteration * 10

        eval = tune_model(model, tokenizer, training_steps, eval_metric, reference_probs, reference_logits, reference_representation)

        model.to("cpu")
        del model, tokenizer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        model = None
        tokenizer = None
        gc.collect()

        return eval
    else:

        total_params_print = print_model_parameters(model, "Total parameters after pruning")

        prune_model_dir = base_dir + '/' + f'prune_iteration_{iteration}'
        
        os.makedirs(prune_model_dir, exist_ok=True)
        model.save_pretrained(prune_model_dir)
        tokenizer.save_pretrained(prune_model_dir)

        model.to("cpu")
        del model, tokenizer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        model = None
        tokenizer = None
        gc.collect()
