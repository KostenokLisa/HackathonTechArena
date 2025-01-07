# Huawei Switzerland Tech Arena - Compression of LLM Inference
This repository contains implementation of the solution which provides **over 4x reduction** in model size with **less than 1% accuracy degradation** through one-shot weight-only quantization.

## Challenge description 
The challenge consists in implementing one-shot compression of the Llama-3.1-8B
model with the goal of achieving the highest compression rate at the lowest
accuracy degradation with respect to the original model. To implement one-shot compression, participants are allowed to implement pruning and
weight-only quantization, but cannot re-train and fine-tune the compressed model to
improve its accuracy. Solution should be integrated in the popular [**lm-evaluation-harness**](https://arxiv.org/abs/2210.17323) benchmarking
framework.

## Approach
The proposed solution utilizes int4 weight quantization for all layers except the Embedding and LM head layers, which are retained in 8-bit precision. To minimize severe accuracy degradation caused by int4 quantization, careful selection of the compression scheme and calibration hyperparameters was critical. The top-scoring submission leverages the state-of-the-art [GPTQ](https://arxiv.org/abs/2210.17323) algorithm and its efficient implementation [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ.git). Compression pipeline is available in `user\compression\compress.py`.

The primary challenge of the task was meeting the time constraints for evaluating the compressed model. Optimized kernels for quantized operations provided by the AutoGPTQ framework were insufficient to achieve a positive score. The only viable solution was to de-quantize the model during runtime and perform testing in bfloat16. Importantly, the formal competition criteria were satisfied, as compression ratios were evaluated based on disk space occupied by checkpoint. I implemented methods for de-quantization of the full model, which are available in the `lm-evaluation-harness\lm_eval\models\user.py`.

## Reproducing results
- Downloading and compressing Llama-3.1-8B checkpoint to 4 bit:
``bash build.sh --compress``
- Downloading 4 bit checkpoint from Hugging Face Hub:
``bash build.sh``
- Testing in benchmarking framework:
``bash run.sh``

