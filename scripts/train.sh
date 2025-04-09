export CUDA_VISIBLE_DEVICES=0,1,2,3  # 指定使用的 GPU
source ~/anaconda3/etc/profile.d/conda.sh
conda activate llm;
python train.py
