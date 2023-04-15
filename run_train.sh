# EmoRoBERTa
#CUDA_VISIBLE_DEVICES=0 python3 run_ememe_main.py \
#    --dataset_name goemotion \
#    --model_name_or_path arpanghoshal/EmoRoBERTa \
#    --do_train \
#    --do_eval \
#    --max_seq_length 128 \
#    --per_device_train_batch_size 32 \
#    --per_device_eval_batch_size 32 \
#    --seed 123 \
#    --learning_rate 1e-5 \
#    --num_train_epochs 3 \
#    --evaluation_strategy epoch \
#    --save_strategy epoch \
#    --load_best_model_at_end \
#    --metric_for_best_model accuracy \
#    --greater_is_better true \
#    --output_dir outputs/test/ \
#    --overwrite_cache

# ViLT
#CUDA_VISIBLE_DEVICES=0 python3 run_ememe_main.py \
#    --dataset_name memotion \
#    --model_name_or_path dandelin/vilt-b32-mlm \
#    --do_train \
#    --do_eval \
#    --max_seq_length 128 \
#    --per_device_train_batch_size 32 \
#    --per_device_eval_batch_size 32 \
#    --seed 123 \
#    --learning_rate 1e-5 \
#    --num_train_epochs 3 \
#    --evaluation_strategy epoch \
#    --save_strategy epoch \
#    --load_best_model_at_end \
#    --metric_for_best_model accuracy \
#    --greater_is_better true \
#    --output_dir outputs/test/ \
#    --overwrite_cache

# Ememe Model
CUDA_VISIBLE_DEVICES=0 python3 run_ememe_main.py \
    --dataset_name ememe \
    --cached_dataset_path data/cached_twitter_data_train.pkl \
    --model_name_or_path ememe \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --seed 123 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model accuracy \
    --greater_is_better true \
    --output_dir outputs/ememe/ \
    --overwrite_cache