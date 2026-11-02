CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python3 master_iterator_multiple_metrics.py --seed 5112 --save_dir ./save_dir_5112 \
    --tuned_original_model  /mnt/sdb/keith/mop_cka_multimodal/output_llava > output_5112.txt 2>&1

CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true python3 llava_alpaca_llm_ft.py --llava_ckpt ./save_dir_5112/prune_iteration_12 \
    --work_dir ./save_dir_5112 \
    --output_llava ./save_dir_5112/fts/it12_5112 \
