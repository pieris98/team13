import trimesh
from signature import SignatureExtractor

import torch, os

faust_remeshed = '/home/lemon/kaist/sem_3/project/new age/diffusion-net/experiments/functional_correspondence/data/faust/off_2'

heat_tensors = []
wave_tensors = []

# Loop through each file in the directory and load it as a tensor
for file_name in sorted(os.listdir(faust_remeshed)):
    file_path = os.path.join(faust_remeshed, file_name)
    print(file_path)
    mesh = trimesh.load(file_path)
    # Compute 100 eigen vectors using laplace-beltrami
    extractor = SignatureExtractor(mesh, 100, approx='beltrami')
    heat_fs, ts = extractor.signatures(64, 'heat', return_x_ticks=True)
    wave_fs = extractor.signatures(128, 'wave')
    heat_tensors.append(heat_fs)
    wave_tensors.append(wave_fs)

# Save the tensors list to a .pt file
torch.save(heat_tensors, "/home/lemon/kaist/sem_3/project/new age/diffusion-net/experiments/functional_correspondence/data/faust/heat_tensors.pt")
torch.save(wave_tensors, "/home/lemon/kaist/sem_3/project/new age/diffusion-net/experiments/functional_correspondence/data/faust/wave_tensors.pt")