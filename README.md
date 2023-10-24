# NanoGPT

NanoGPT is a simple character-level GPT model that can easily be trained on any .txt dataset you provide. The larger the better! This model performs (best case) on-par with GPT-2.

## Install Dependencies
```
pip install numpy transformers datasets tiktoken wandb tqdm
```
IF you have a GPU (and would like to use it), install CUDA 12.1 (and cuDNN) and then run this command to get the compatible version of Pytorch:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
If you do NOT have a GPU you would like to use:
```
pip install torch
```

# Quick Start
1. Store your dataset in the 'datasets' directory (or use one of the existing datasets), and provide the filename in data\dataset_char\prepare.py
2. To prepare/tokenize the dataset, assign the path of the dataset to the input_file_path variable in prepare.py and run it:
```
python data/dataset_char/prepare.py
```
3. To train the model on your dataset (expect this to run for 2-6 minutes), run:
```
python train.py config/train_dataset_char.py
```
4. To view the results, run:
```
python sample.py --out_dir=out-dataset-char
```

## Tools
* collect_clean_lyrics: This directory contains scripts that can be used to collect the lyrical discography of an artist and clean the data. Read collect_clean_lyrics\README.md for more info. 
* tools\testCUDA.py: This script can tell you if you have a GPU at your disposal. The training will be completed significantly faster if you have a GPU.

## Sources
* I learned how to build this model thanks to Andrej Karpathy's (OpenAI Cofounder) fantastic video: https://www.youtube.com/watch?v=kCc8FmEb1nY

Datasets:
* shakespeare.txt: All of Shakespeare's work, which can be downloaded at: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
* travis_scott.txt: All of Travis Scott's lyrics (as of October 10, 2023)
* trump_tweets.txt: All of President Trump's tweets (excluding retweets)