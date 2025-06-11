import torch
print(torch.__version__, "CUDA", torch.version.cuda, "GPU?", torch.cuda.is_available())
