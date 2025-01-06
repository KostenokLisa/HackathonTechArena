import numpy as np
import torch
import torch.nn as nn
import random
from torch import save
from os import getenv
import transformers
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from bitsandbytes.nn import Embedding8bit, Linear8bitLt

pretrained_model_dir = "meta-llama/Llama-3.1-8B"
quantized_model_dir = "llama-gptq-bnb"

def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp.to("cuda:0"), "attention_mask": attention_mask.to("cuda:0")})
    return traindataset


def main():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
    traindataset = get_wikitext2(128, 0, 2048, tokenizer)

    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False,  # desc_act and group size only works on triton
    )
    torch_dtype = torch.bfloat16
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config, device_map="cuda:0",  low_cpu_mem_usage=True, torch_dtype=torch_dtype)
    print(model)
    model = model.to("cuda:0")
    model.quantize(traindataset, use_triton=False)
    env=getenv("OUTPUT_MODEL")
    ckpt_path = f"{env}/{quantized_model_dir}.pt"
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            fp16_embedding = module
            int8_embedding = Embedding8bit(module.weight.size(0), module.weight.size(1), dtype=torch_dtype)
            int8_embedding.load_state_dict(fp16_embedding.state_dict())
            int8_embedding = int8_embedding.to("cuda:0")
            parent_module = dict(model.named_modules())[name.rsplit('.', 1)[0]] if '.' in name else model
            setattr(parent_module, name.split('.')[-1], int8_embedding)

        elif "lm_head" in name:
            fp16_lm_head = module
            int8_lm_head = Linear8bitLt(module.weight.size(1), module.weight.size(0), bias=False, has_fp16_weights=False)
            int8_lm_head.load_state_dict(fp16_lm_head.state_dict())
            int8_lm_head = int8_lm_head.to("cuda:0")
            parent_module = dict(model.named_modules())[name.rsplit('.', 1)[0]] if '.' in name else model.model
            setattr(parent_module, name.split('.')[-1], int8_lm_head)

    ckpt_path_w_emb = f"{env}/{quantized_model_dir}_w_emb.pt"
    save(model, ckpt_path_w_emb)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()