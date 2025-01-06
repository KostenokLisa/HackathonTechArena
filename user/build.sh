#!/bin/bash 
set -x
set -e
# loging to HF
huggingface-cli login --token ${HF_TOKEN}
# install dependencies specified in setup.py
python3 setup.py install
yum install -y wget
yum install -y python3-devel
yum install -y gcc g++
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sh cuda_12.2.0_535.54.03_linux.run --silent --toolkit
git clone https://github.com/PanQiWei/AutoGPTQ.git
export CUDA_HOME=/usr/local/cuda-12.2
ls /usr/local/cuda-12.2
cd AutoGPTQ
pip install wheel
pip install -vvv --no-build-isolation -e .
cd ..
huggingface-cli download KostenokLisa/llama-gptq-bnb-bf16 llama-gptq-bnb_w_emb.pt --local-dir ${OUTPUT_MODEL}
cd compression/output_model
rm -rf .cache
cd ../..
if [[ $1 == "--compress" ]]; then
	for i in {0..10};do echo "";done
	echo "Put here the steps needed to compress the full precision model"
	echo "Any dependencies needed must be specified in setup.py"
	echo "The source files with the compression code need to be stored in compress"
	echo "At the end of this script, all the files needed to LOAD the model for evaluation must be stored in $OUTPUT_MODEL"
	cd compression/output_model
	rm -rf llama-gptq-bnb_w_emb.pt
	cd ..
	python3 compress.py
fi
echo "BUILD SUCCESSFUL"
