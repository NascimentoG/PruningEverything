env CUDA_VISIBLE_DEVICES=0 python3 master_iterator_multiple_metrics.py \ 
    --metric cka \
    --num_iterations 18 \
    --seed 0 \
    --save_dir "./save_dir" \

env CUDA_VISIBLE_DEVICES=0 python3 fine_tuning.py \
    --pruned_model_dir ./save_dir/prune_iteration_15 \
    --tuned_model_dir ./save_dir/15_pos_ft_seed_0 \