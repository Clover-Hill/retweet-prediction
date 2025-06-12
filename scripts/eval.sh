DATASET=/fs-computility/plm/shared/jqcao/projects/retweet-prediction/data/preprocessed_dataset

# ACCELERATE_CONFIG=./accelerate_config/eval.yaml
ACCELERATE_CONFIG=./accelerate_config/train-8-card.yaml
MODEL=/fs-computility/plm/shared/jqcao/projects/retweet-prediction/checkpoint/all-classification-epoch-50/epoch_5
OUTPUT_DIR=./result/classification_only

accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    -m \
    eval \
    --classification_model_path ${MODEL} \
    --dataset_name ${DATASET} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_eval_batch_size 32 \
    --do_test \
    --viral_threshold 0.86 \
    --viral_mean_val 50 \
    --report_to none