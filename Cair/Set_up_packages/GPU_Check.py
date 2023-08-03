
import torch

output_file = r'/gpfs/home/hluo/cuda_check.txt'

python_version = sys.version_info


with open(output_file, 'w') as file:
    if torch.cuda.is_available():
        print(torch.cuda.device_count(),file=file)
        print("CUDA is available./n", file=file)
    else:
        print("CUDA is not available.", file=file)
        print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}",file=file)


print("Output saved to:", output_file)
