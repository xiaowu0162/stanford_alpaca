export CUDA_VISIBLE_DEVICES=4

NGPUS=1

torchrun --nproc_per_node=${NGPUS} --master_port=5633 train.py \
	 --model_name_or_path ./llama_checkpoint/7B/llama-7b/ \
	 --data_path ./alpaca_data.json \
	 --bf16 True \
	 --output_dir checkpoint_test \
	 --num_train_epochs 3 \
	 --per_device_train_batch_size 1 \
	 --per_device_eval_batch_size 1 \
	 --gradient_accumulation_steps 128 \
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
