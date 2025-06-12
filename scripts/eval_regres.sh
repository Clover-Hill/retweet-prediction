DATASET=/fs-computility/plm/shared/jqcao/projects/retweet-prediction/data/preprocessed_dataset

# ACCELERATE_CONFIG=./accelerate_config/eval.yaml
ACCELERATE_CONFIG=./accelerate_config/train-4-card.yaml
REGRESSION_MODEL=/fs-computility/plm/shared/jqcao/projects/retweet-prediction/checkpoint/all-regression-epoch-10/epoch_1
OUTPUT_DIR=./result/all_regression_epoch1

accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    -m \
    eval_regres \
    --regression_model_path ${REGRESSION_MODEL} \
    --dataset_name ${DATASET} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_eval_batch_size 32 \
    --do_test \
    --report_to none