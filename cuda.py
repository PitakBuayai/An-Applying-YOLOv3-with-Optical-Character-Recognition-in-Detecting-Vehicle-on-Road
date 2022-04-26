
import torch

x = torch.rand(5, 3)
print(x)


device_id = torch.cuda.current_device()
gpu_properties = torch.cuda.get_device_properties(device_id)
 