export CUDA_VISIBLE_DEVICES=2,3

NGPUS=2

#     --fsdp_transformer_layer_cls_to_wrap 'LLaMADecoderLayer' \
#	 --fsdp "full_shard auto_wrap" \

dataset=kp20k-20k-1

N_EPOCHS=3
# SAVE_STEPS=250
LR=1e-5
WARMUP=0.03
BSZ=8
GRAD_ACC=4

torchrun --nproc_per_node=${NGPUS} --master_port=5633 train.py \
     --model_name_or_path ./vicuna-7b-v1.1/ \
     --data_path ./keyphrase_data/${dataset}/train.json \
     --fp16 True \
     --deepspeed stage3.config \
     --bf16 False \
     --output_dir 20220421_checkpoint_vicuna_${dataset}_${N_EPOCHS}epochs_lr${LR}_warmup${WARMUP}_bsz${NGPUS}x${BSZ}x${GRAD_ACC} \
     --num_train_epochs ${N_EPOCHS} \
     --per_device_train_batch_size ${BSZ} \
     --per_device_eval_batch_size 1 \
     --gradient_accumulation_steps ${GRAD_ACC} \
     --evaluation_strategy "no" \
     --save_strategy "epoch" \
     --save_total_limit 1 \
     --learning_rate ${LR} \
     --weight_decay 0. \
     --warmup_ratio ${WARMUP} \
     --lr_scheduler_type "cosine" \
     --logging_steps 1 \
     --tf32 False


# from Wade
#python -u train.py --model_name_or_path ./llama_checkpoint/7B/llama-7b/ --data_path ./alpaca_data.json --fp16 True --bf16 False --output_dir checkpoint_test --num_train_epochs 2 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 4 --evaluation_strategy no --save_strategy steps --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine --logging_steps 1 --deepspeed stage3.config --tf32 False


# from https://github.com/tatsu-lab/stanford_alpaca/issues/46
# torchrun --nproc_per_node=8 --master_port=1234 train.py --model_name_or_path converted_llama_7B --data_path ./alpaca_data.json --fp16 True --output_dir ./trained_model --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
