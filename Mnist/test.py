import torch
# 测试CUDA
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print(torch.cuda.get_device_name(0))
else:
    print('cpu')