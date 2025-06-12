DATASET=/fs-computility/plm/shared/jqcao/projects/retweet-prediction/data/preprocessed_dataset
NUM_EPOCH=50
LR="5e-4"

ACCELERATE_CONFIG=./accelerate_config/train-8-card.yaml
MODEL=/fs-computility/plm/shared/jqcao/models/Qwen3/Qwen3-Embedding-0.6B
OUTPUT_DIR=/fs-computility/plm/shared/jqcao/projects/retweet-prediction/checkpoint/viral-multi_regression-epoch-${NUM_EPOCH}

WANDB_PROJECT="retweet-prediction" accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    -m \
    train \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET} \
    --output_dir ${OUTPUT_DIR} \
    --learning_rate ${LR} \
    --lr_scheduler_type linear \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs ${NUM_EPOCH} \
    --head_type "multi_regression" \
    --mlp_num 10 \
    --scalar_features_dim 7 \
    --dropout_rate 0.1 \
    --neg_pos_ratio 5.67 \
    --num_class 16 \
    --seed 42 \
    --checkpointing_steps "epoch" \
    --report_to none
    # --with_tracking \
    # --project_name retweet-prediction \
    # --run_name all-regression-epoch-${NUM_EPOCH}