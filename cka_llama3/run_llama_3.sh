env CUDA_VISIBLE_DEVICES=0 python3 master_iterator_multiple_metrics.py \
    --metric cka \
    --num_iterations 20 \
    --seed 42 \
    --save_dir "./save_dir/" \

env CUDA_VISIBLE_DEVICES=0 python3 fine_tuning.py --pruned_model_dir ./save_dir/prune_iteration_13 \
    --tuned_model_dir ./save_dir/13_pos_ft_seed_333 \
