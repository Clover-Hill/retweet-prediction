DATASET=/fs-computility/plm/shared/jqcao/projects/retweet-prediction/data/feature_dataset
NUM_EPOCH=10
LR="1e-4"

ACCELERATE_CONFIG=./accelerate_config/train-8-card.yaml
# ACCELERATE_CONFIG=./accelerate_config/eval.yaml
MODEL=/fs-computility/plm/shared/jqcao/models/Qwen3/Qwen3-Embedding-0.6B
OUTPUT_DIR=/fs-computility/plm/shared/jqcao/projects/retweet-prediction/checkpoint/all-classification-epoch-${NUM_EPOCH}-${LR}

WANDB_PROJECT="retweet-prediction" accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    --main_process_port 29501 \
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
    --head_type "classification" \
    --mlp_num 10 \
    --dropout_rate 0.1 \
    --neg_pos_ratio 5.67 \
    --seed 42 \
    --checkpointing_steps "epoch" \
    --report_to none
    # --with_tracking \
    # --project_name retweet-prediction \
    # --run_name all-classification-epoch-${NUM_EPOCH}-${LR}