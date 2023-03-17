export CUDA_VISIBLE_DEVICES=4,5

NGPUS=2

torchrun --nproc_per_node=${NGPUS} --master_port=5633 train.py \
	 --model_name_or_path ./llama_checkpoint/7B/llama-7b/ \
	 --data_path ./alpaca_data.json \
	 --bf16 True \
	 --output_dir checkpoint_test \
	 --num_train_epochs 3 \
	 --per_device_train_batch_size 1 \
	 --per_device_eval_batch_size 1 \
	 --gradient_accumulation_steps 64 \
	 --evaluation_strategy "no" \
	 --save_strategy "steps" \
	 --save_steps 2000 \
	 --save_total_limit 1 \
	 --learning_rate 2e-5 \
	 --weight_decay 0. \
	 --warmup_ratio 0.03 \
	 --lr_scheduler_type "cosine" \
	 --logging_steps 1 \
	 --fsdp "full_shard auto_wrap" \
	 --fsdp_transformer_layer_cls_to_wrap 'LLaMADecoderLayer' \
	 --tf32 True

# from https://github.com/tatsu-lab/stanford_alpaca/issues/46
# torchrun --nproc_per_node=8 --master_port=1234 train.py --model_name_or_path converted_llama_7B --data_path ./alpaca_data.json --fp16 True --output_dir ./trained_model --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "steps" --save_steps 2000 --save_total_limit 1 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
