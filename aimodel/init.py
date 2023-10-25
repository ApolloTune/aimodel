import os
import torch
def startupInit():
    for dirname, _, filenames in os.walk('../dataset'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("We will use the GPU: ",torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using the CPU instead")
        device = torch.device("cpu")
        print(device)