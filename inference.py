from dataclasses import dataclass, field
import os
import json
from tqdm import tqdm
import numpy as np
import torch
import transformers
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM

from train import ModelArguments, smart_tokenizer_and_embedding_resize, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, \
  DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, PROMPT_DICT


@dataclass
class InferenceArguments:
  model_max_length: int = field(
    default=512,
    metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
  )
  load_in_8bit: bool = field(
    default=False,
    metadata={"help": "Load the model in 8-bit mode."},
  )
  inference_dtype: torch.dtype = field(
    default=torch.float32,
    metadata={"help": "The dtype to use for inference."},
  )


def generate_prompt(instruction, input=None):
  if input:
    return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
  else:
    return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


def inference():
  parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
  model_args, inference_args = parser.parse_args_into_dataclasses()

  model = LlamaForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    load_in_8bit=inference_args.load_in_8bit,
    torch_dtype=torch.float16,
    device_map="auto",
  )
  model.cuda()
  model.eval()

  generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    num_beams=4,
  )

  tokenizer = LlamaTokenizer.from_pretrained(
    model_args.model_name_or_path,
    use_fast=False,
    model_max_length=inference_args.model_max_length,
  )

  if tokenizer.pad_token is None:
    smart_tokenizer_and_embedding_resize(
      special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
      tokenizer=tokenizer,
      model=model,
    )
  tokenizer.add_special_tokens(
    {
      "eos_token": DEFAULT_EOS_TOKEN,
      "bos_token": DEFAULT_BOS_TOKEN,
      "unk_token": DEFAULT_UNK_TOKEN,
    }
  )

  with open("/local2/wadeyin/dynamic_inst/DynInstruct/data/splits/default/hf_train_continue_0_681_new_distribution_llama/test_tasks.txt") as f:
    tasks = [t.strip() for t in f]

  instruction_data = []
  for t in tqdm(tasks):
    with open(os.path.join("/local2/wadeyin/dynamic_inst/DynInstruct/data/tasks", t+'.json'), 'r') as f:
      test_data = json.load(f)
    for test_d in test_data["Instances"][:100]:
      d = dict()
      d["instruction"] = test_data["Definition"][0]
      d["input"] = test_d["input"]
      d["output"] = test_d["output"]
      d["task"] = t

      instruction_data.append(d)

    with open("niv2_test_data.json", "w") as f:
      json.dump(instruction_data, f, indent=4)

  with open("niv2_test_outputs_zeroshot.json", "w") as f:
    for instruction in tqdm(instruction_data):
      inputs = tokenizer(generate_prompt(instruction["instruction"], instruction["input"]), return_tensors="pt")
      outputs = model.generate(input_ids=inputs["input_ids"].cuda(),
                               generation_config=generation_config,
                               max_new_tokens=inference_args.model_max_length,
                               return_dict_in_generate=True,
                               output_scores=True)

      input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
      generated_tokens = outputs.sequences[:, input_length:]

      tmp = instruction.copy()
      tmp["pred"] = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
      f.write(json.dumps(tmp)+'\n')

if __name__ == "__main__":
  inference()