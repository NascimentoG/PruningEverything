from typing import List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
import random
import gc
from typing import List, Optional, Tuple, Sequence
from datasets import load_dataset, Image as HFImage, DatasetDict

LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

PRE_PROMPT = ""
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."


def build_mc_prompt(
    question: str,
    choices: List[str],
    hint: Optional[str],) -> str:
    """
    Build a multiple-choice prompt for LLaVA.
    """
    choices_str = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))

    ctx = f"Context: {hint}\n" if hint and str(hint).strip() else ""
    return f"{PRE_PROMPT}{ctx}{question}\n{choices_str}{POST_PROMPT}"


def cast_image_column(ds):
    """
    Cast the image column to HFImage type.
    """
    if "image" in ds.column_names and not isinstance(ds.features["image"], HFImage):
        ds = ds.cast_column("image", HFImage())
    return ds

def has_image(example):
    """
    Returns True if the example has an image.
    """
    return example.get("image") is not None

def preprocess_example(example, processor):
    """
    Preprocess a single example for LLaVA fine-tuning. Returns input_ids, labels, and pixel_values.
    """
    tokenizer = processor.tokenizer

    question = example["question"]
    choices = example["choices"]
    hint = example.get("hint", None)

    ans_idx = int(example["answer"]) if not isinstance(example["answer"], list) else int(example["answer"][0])
    gold_letter = LETTER_MAP.get(ans_idx, "A")

    image = example.get("image", None)
    mc_text = build_mc_prompt(question, choices, hint)

    conversation = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": mc_text}],
    }]

    text_prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    enc = processor(text=[text_prompt], images=[image], return_tensors="pt")
    input_ids = enc["input_ids"][0]
    pixel_values = enc["pixel_values"][0]

    answer_ids = tokenizer(gold_letter, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    answer_ids = torch.tensor(answer_ids, dtype=torch.long)

    input_ids_full = torch.cat([input_ids, answer_ids], dim=0)

    labels = torch.full_like(input_ids_full, fill_value=-100)
    labels[-len(answer_ids):] = answer_ids

    out = {
        "input_ids": input_ids_full.tolist(),
        "labels": labels.tolist(),
        "pixel_values": pixel_values.numpy(),
    }
    return out

def extract_representations_from_multimodal_model(
        model,
        tokenizer,
        train_dataset,
        device,
        batch_size = 1,
        layers_to_extract = [-1],
        feature_selection = "last_token") -> Tuple[torch.Tensor,...]:
    
    all_features = {layer: [] for layer in layers_to_extract}

    for i in range(0,len(train_dataset), batch_size):
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in train_dataset.select(range(i,i + batch_size))]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in train_dataset.select(range(i,i + batch_size))]
        max_len = max(x.size(0) for x in input_ids)

        def pad(seq_list, pad_value):
            out = torch.full((len(seq_list), max_len), pad_value, dtype=seq_list[0].dtype)
            for i, s in enumerate(seq_list):
                out[i, : s.size(0)] = s
            return out

        batch_input_ids = pad(input_ids, tokenizer.pad_token_id)
        batch_labels = pad(labels, -100)
        batch_attn = (batch_input_ids != tokenizer.pad_token_id).long()

        pixel_values = torch.stack([torch.tensor(f["pixel_values"]) for f in train_dataset.select(range(i,i + batch_size))], dim=0)

        with torch.no_grad():
            model_outputs = model(
                input_ids=batch_input_ids.to(device),
                pixel_values=pixel_values.to(device),
                labels=batch_labels.to(device),
                attention_mask=batch_attn.to(device),
                output_hidden_states=True
            )
        


        for layer in layers_to_extract:
                if feature_selection == "last_token":
                    hidden_states_layer = model_outputs.hidden_states[layer] 
                    
                    sequence_lengths = batch_attn.sum(dim=1)
                    
                    last_token_indices = sequence_lengths - 1
                    
                    last_token_activations = []
                    for i in range(hidden_states_layer.size(0)): 
                        last_idx = last_token_indices[i]
                        last_token_activations.append(hidden_states_layer[i, last_idx, :])
                    
                    activations = torch.stack(last_token_activations, dim=0)
                all_features[layer].append(activations.cpu())
        
        del input_ids, batch_attn, batch_input_ids, pixel_values, model_outputs
        torch.cuda.empty_cache()  
        
    for layer in layers_to_extract:
        all_features[layer] = torch.cat(all_features[layer], dim=0)

    return all_features

def get_processed_scienceqa_samples(num_samples=128, processor  = None):
    """
    Get random samples from wikitext2 dataset.
    Randomly samples until we get the specified number of non-empty samples.
    """
    dataset = load_dataset("derek-thomas/ScienceQA", split="train")
    ds_train = cast_image_column(dataset).filter(has_image)

    subset = []
    def _map_fn(ex): return preprocess_example(ex, processor)
    keep_cols = ("image", "question", "choices", "answer", "hint")
    ds_train_pp = ds_train.map(_map_fn, remove_columns=[c for c in ds_train.column_names if c in keep_cols])

    ds_train_pp = ds_train_pp.shuffle().select(range(num_samples))

    return ds_train_pp


def feature_space_linear_cka(
    features_x: torch.Tensor,
    features_y: torch.Tensor
) -> torch.Tensor:
    """
    Compute linear CKA between two feature matrices (n×d), optionally debiased.
    """
    features_x = features_x - features_x.mean(dim=0, keepdim=True)
    features_y = features_y - features_y.mean(dim=0, keepdim=True)

    k_xy = features_x.t().mm(features_y)       
    k_xx = features_x.t().mm(features_x)       
    k_yy = features_y.t().mm(features_y)       

    dot_xy = torch.norm(k_xy) ** 2             
    norm_x = torch.norm(k_xx)                  
    norm_y = torch.norm(k_yy)                  

    cka = 1-(dot_xy / (norm_x * norm_y))
    dissimilarity = cka.detach().cpu().item()
    return dissimilarity


def compute_model_representation(model, processor, device, num_samples=128): 
    """
    Runs the model on random samples from wikitext2,
    applies softmax to the logits, and returns the probability distributions.
    """
    model.eval()
    processed_samples = get_processed_scienceqa_samples(num_samples,processor)

    representations = extract_representations_from_multimodal_model(model, processor.tokenizer, processed_samples, device, batch_size=1, layers_to_extract=[-1], feature_selection="last_token")

    gc.collect()

    return representations[-1] 


def compute_cka_from_reprs(repr_ref, repr_pruned):
    """
    Computes the CKA between two representations.
    """
    print(f"Reference representation shape: {repr_ref.shape}")
    print(f"Pruned representation shape: {repr_pruned.shape}")

    if repr_ref.shape != repr_pruned.shape:
        print("Shapes don't match, adjusting dimensions...")
        min_seq = min(repr_ref.shape[1], repr_pruned.shape[1])

        repr_ref = repr_ref[:, :min_seq, :]
        repr_pruned = repr_pruned[:, :min_seq, :]
        print(f"Adjusted shapes - Ref: {repr_ref.shape}, Pruned: {repr_pruned.shape}")

    cka_dissimilarity = feature_space_linear_cka(repr_ref, repr_pruned)
    del repr_pruned
    gc.collect()
    return cka_dissimilarity
