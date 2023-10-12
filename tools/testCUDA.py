# This script enables you to check if you have a GPU available
# If you have a GPU available (and you correctly installed CUDA & cuDNN) you will see "cuda" and the cuda version available
# If you do not have a GPU available (or you failed to install CUDA & cuDNN) you will see "cpu" and "none"

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(torch.version.cuda)
