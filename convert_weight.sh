
out_dir=./llama_checkpoint/7B
mkdir -p $out_dir

python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /local/llama/model/ \
    --model_size 7B \
    --output_dir $out_dir

cp ./llama_checkpoint/tokenizer/* ${out_dir}

echo "Note: there is a recent change in alpaca/llama repo. You should check the configs and replace Llama with LLaMA in the model and tokenizer types!"
