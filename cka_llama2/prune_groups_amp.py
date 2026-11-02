import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import random
import numpy as np
import pickle
import os
import gc
import argparse
from amp_criterion import measure_amp_heads_importance, measure_amp_mlps_importance, remove_all_hooks
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


def prune_model(model, tokenizer, final_pruning_ratio):
    """
    Prunes the model's attention heads and MLP units based on the specified pruning_ratio.
    The pruning is done using AMP (Activation Magnitude Pruning) to determine importance.

    Args:
        model (nn.Module): The language model to be pruned.
        final_pruning_ratio (float): The fraction of total parameters to prune (between 0 and 1).

    Returns:
        nn.Module: The pruned model.
    """
    total_attention_params = 0
    total_mlp_params = 0

    for layer in model.model.layers:
        total_attention_params += sum(p.numel() for p in layer.self_attn.parameters())
        total_mlp_params += sum(p.numel() for p in layer.mlp.parameters())

    total_params = total_attention_params + total_mlp_params

    actual_total_params = sum(p.numel() for p in model.parameters())

    pruning_ratio = (actual_total_params * final_pruning_ratio) / (total_params)

    amp_head_importances = measure_amp_heads_importance(
        model,
        tokenizer,
        arch="llama",
        dataset_name="yahma/alpaca-cleaned",
        split="train",
        max_prompts=128,
        random_subset=True
    )
    remove_all_hooks()

    amp_mlps_importance = measure_amp_mlps_importance(
        model,
        tokenizer,
        arch="llama",
        dataset_name="yahma/alpaca-cleaned",
        split="train",
        max_prompts=128,
        random_subset=True
    )
    remove_all_hooks()

    for layer_num, layer in enumerate(model.model.layers):
        num_heads = layer.self_attn.num_heads
        num_prune_heads = round(pruning_ratio * num_heads)

        if num_prune_heads >= num_heads:
            num_prune_heads = num_heads - 1

        layer_heads_importance = amp_head_importances[layer_num]
        sorted_heads_indices = np.argsort(layer_heads_importance)

        heads_to_prune = sorted_heads_indices[:num_prune_heads]
        prune_attention_heads(layer.self_attn, heads_to_prune)

        intermediate_size = layer.mlp.gate_proj.out_features
        num_prune_mlp_units = round(intermediate_size*((num_heads / intermediate_size) * (pruning_ratio - (num_prune_heads / num_heads)) + pruning_ratio))

        if num_prune_mlp_units >= intermediate_size:
            num_prune_mlp_units = intermediate_size - 1
        
        layer_mlps_importance = amp_mlps_importance[layer_num]
        sorted_mlps_indices = np.argsort(layer_mlps_importance)

        mlp_units_to_prune = sorted_mlps_indices[:num_prune_mlp_units]
        prune_mlp_units(layer.mlp, mlp_units_to_prune)

    return model


def prune_attention_heads(attention_layer, heads_to_prune):
    """
    Prunes specified attention heads in the attention layer.

    Args:
        attention_layer (nn.Module): The attention layer to prune.
        heads_to_prune (list): Indices of the heads to prune.
    """
    num_heads = attention_layer.num_heads
    head_dim = attention_layer.head_dim

    heads_to_keep = sorted(set(range(num_heads)) - set(heads_to_prune))

    idxs_to_keep = []
    for head in heads_to_keep:
        idxs = list(range(head * head_dim, (head + 1) * head_dim))
        idxs_to_keep.extend(idxs)
    idxs_to_keep = sorted(idxs_to_keep)

    attention_layer.q_proj.weight = torch.nn.Parameter(
        attention_layer.q_proj.weight.data[idxs_to_keep, :]
    )

    attention_layer.k_proj.weight = torch.nn.Parameter(
        attention_layer.k_proj.weight.data[idxs_to_keep, :]
    )

    attention_layer.v_proj.weight = torch.nn.Parameter(
        attention_layer.v_proj.weight.data[idxs_to_keep, :]
    )

    attention_layer.o_proj.weight = torch.nn.Parameter(
        attention_layer.o_proj.weight.data[:, idxs_to_keep]
    )

    attention_layer.num_heads = len(heads_to_keep)
    attention_layer.hidden_size = attention_layer.num_heads * attention_layer.head_dim

    attention_layer.q_proj.out_features = attention_layer.hidden_size
    attention_layer.k_proj.out_features = attention_layer.hidden_size
    attention_layer.v_proj.out_features = attention_layer.hidden_size
    attention_layer.o_proj.in_features = attention_layer.hidden_size

    return


def prune_mlp_units(mlp_layer, units_to_prune):
    """
    Prunes specified units in the MLP layer.

    Args:
        mlp_layer (nn.Module): The MLP layer to prune.
        units_to_prune (list): Indices of the units to prune.
    """
    keep_idxs = sorted(set(range(mlp_layer.gate_proj.out_features)) - set(units_to_prune))

    mlp_layer.gate_proj.out_features = len(keep_idxs)
    mlp_layer.gate_proj.weight = torch.nn.Parameter(
        mlp_layer.gate_proj.weight.data[keep_idxs, :]
    )

    mlp_layer.up_proj.out_features = len(keep_idxs)
    mlp_layer.up_proj.weight = torch.nn.Parameter(
        mlp_layer.up_proj.weight.data[keep_idxs, :]
    )

    mlp_layer.down_proj.in_features = len(keep_idxs)
    mlp_layer.down_proj.weight = torch.nn.Parameter(
        mlp_layer.down_proj.weight.data[:, keep_idxs]
    )

    return


def prune_groups(args, iteration, base_dir, eval_metric, tune_eval=False, reference_probs=None, reference_logits=None, reference_representation=None):
    """
    Main function to load the model, perform pruning, update the configuration, and save the pruned model.
    """
    torch.cuda.empty_cache()
    gc.collect()
    
    config = AutoConfig.from_pretrained(args.model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"Warning during model loading: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch.float16,
            ignore_mismatched_sizes=True
        )

    print("Loading tokenizer from tuned original model...")
    tokenizer = AutoTokenizer.from_pretrained(args.tuned_original_tokenizer)
    
    tokenizer.pad_token = tokenizer.eos_token  
    print("Tokenizer loaded from tuned original model.")

    torch.cuda.empty_cache()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Moving model to device: {device}")
    model.to(device)

    total_params = print_model_parameters(model, "Total parameters before pruning")
    params_to_prune = sum(p.numel() for p in model.model.layers[0].parameters())

    final_pruning_ratio = params_to_prune / total_params
    print("Iteration pruning ratio: ", final_pruning_ratio)
    model = prune_model(model, tokenizer, final_pruning_ratio)

    new_num_heads = model.model.layers[0].self_attn.num_heads
    model.config.num_attention_heads = new_num_heads
    model.config.num_key_value_heads = new_num_heads
    model.config.head_dim = 128

    new_intermediate_size = model.model.layers[0].mlp.gate_proj.out_features
    model.config.intermediate_size = new_intermediate_size

    for layer in model.model.layers:
        layer.self_attn.num_key_value_heads = new_num_heads

    total_params_print = print_model_parameters(model, "Total parameters after pruning")

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
        print(f"Eval groups: {eval}")
        return eval
    else:
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