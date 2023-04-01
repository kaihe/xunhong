TOT_CUDA="0,1,2,3,4,5,6,7"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="1234"

DATA_PATH="./train_data/belle_metra_diago.json"
OUTPUT_PATH="./adapters/belle+metra"
MODEL_PATH="decapoda-research/llama-7b-hf"

CUDA_VISIBLE_DEVICES=${TOT_CUDA} torchrun --nproc_per_node=$CUDA_NUM --master_port=$PORT finetune.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 200 \
--save_steps 200 \
--test_size 1