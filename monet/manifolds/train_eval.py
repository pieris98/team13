import time
import torch
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

from matplotlib import pyplot as plt

import pygeodesic.geodesic as geodesic
import trimesh
import sys

def print_info(info):
    message = ('Epoch: {}/{}, Duration: {:.3f}s, ACC: {:.4f}, '
               'Train Loss: {:.4f}, Test Loss:{:.4f}').format(
                   info['current_epoch'], info['epochs'], info['t_duration'],
                   info['acc'], info['train_loss'], info['test_loss'])
    print(message)


def run(model, train_loader, test_loader, target, num_nodes, epochs, optimizer,
        scheduler, device):
    
    best_test_loss = float('inf')
    best_model_state_dict = None

    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss = train(model, train_loader, target, optimizer, device)
        t_duration = time.time() - t
        scheduler.step()
        acc, test_loss = test(model, test_loader, num_nodes, target, device)

        if test_loss < best_test_loss: 
            best_test_loss = test_loss
            best_model_state_dict = model.state_dict()
            best_model = model


        eval_info = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'acc': acc,
            'current_epoch': epoch,
            'epochs': epochs,
            't_duration': t_duration
        }

        print_info(eval_info)

        if best_model_state_dict is not None:
            torch.save(best_model_state_dict, f"monet_{test_loss}.pth")

        hardcoded_filename = "benchmark_results"
        gt_dir = "/home/lemon/kaist/sem_3/project/monet/data/FAUST/raw/MPI-FAUST/test/scans/test_scan_000.ply"
        princeton_benchmark(best_model, test_loader, gt_dir, hardcoded_filename)

def train(model, train_loader, target, optimizer, device):
    model.train()

    total_loss = 0
    for idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        loss = F.nll_loss(model(data.to(device)), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(model, test_loader, num_nodes, target, device):
    model.eval()
    correct = 0
    total_loss = 0
    n_graphs = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            out = model(data.to(device))
            total_loss += F.nll_loss(out, target).item()
            pred = out.max(1)[1]
            correct += pred.eq(target).sum().item()
            n_graphs += data.num_graphs
    return correct / (n_graphs * num_nodes), total_loss / len(test_loader)

def normalize_mesh(mesh):
    """Center mesh and scale x, y and z dimension with '1/geodesic diameter'.

    Parameters
    ----------
    mesh: trimesh.Trimesh
        The triangle mesh, that shall be normalized

    Returns
    -------
    trimesh.Trimesh:
        The normalized mesh
    """
    # Center mesh
    for dim in range(3):
        mesh.vertices[:, dim] = mesh.vertices[:, dim] - mesh.vertices[:, dim].mean()

    # Determine geodesic distances
    n_vertices = mesh.vertices.shape[0]
    distance_matrix = np.zeros((n_vertices, n_vertices))
    geoalg = geodesic.PyGeodesicAlgorithmExact(mesh.vertices, mesh.faces)
    for sp in tqdm(range(n_vertices), postfix=f"Normalizing mesh.."):
        distances, _ = geoalg.geodesicDistances([sp], None)
        distance_matrix[sp] = distances

    # Scale mesh
    geodesic_diameter = distance_matrix.max()
    for dim in range(3):
        mesh.vertices[:, dim] = mesh.vertices[:, dim] * (1 / geodesic_diameter)
    print(f"-> Normalized with geodesic diameter: {geodesic_diameter}")

    return mesh

def princeton_benchmark(model, test_loader, ref_mesh_path, file_name):
    """Plots the accuracy w.r.t. a gradually changing geodesic error

    Princeton benchmark has been introduced in:
    > [Blended intrinsic maps](https://doi.org/10.1145/2010324.1964974)
    > Vladimir G. Kim, Yaron Lipman and Thomas Funkhouser

    Parameters
    ----------
    model: torch.nn.Module
        The monet trained model
    test_dataset: torch.nn.data.Dataset
        The test dataset on which to evaluate the MoNet
    ref_mesh_path: str
        A path to the reference mesh
    file_name: str
        The file name under which to store the plot and the data (without file format ending!)
    """



    reference_mesh = trimesh.load_mesh(ref_mesh_path)

    for idx, data in enumerate(test_loader):

    # reference_mesh = trimesh.load_mesh(ref_mesh_path)
    # reference_mesh = normalize_mesh(reference_mesh)
    # geoalg = geodesic.PyGeodesicAlgorithmExact(reference_mesh.vertices, reference_mesh.faces)

    # geodesic_errors, mesh_idx = [], -1
    # for data in test_loader:
    #     mesh_idx += 1
    #     signal, barycentric, ground_truth = data.signal, data.barycentric, data.ground_truth
    #     prediction = model([signal, barycentric]).numpy().argmax(axis=1)
    #     pred_idx = -1
    #     for gt, pred in np.stack([ground_truth, prediction], axis=1):
    #         pred_idx += 1
    #         sys.stdout.write(f"\rCurrently at mesh {mesh_idx} - Prediction {pred_idx}")
    #         geodesic_errors.append(geoalg.geodesicDistance(pred, gt)[0])
    #     break

    # geodesic_errors = np.array(geodesic_errors)
    # geodesic_errors.sort()

    # ###########
    # # Plotting
    # ###########
    # amt_values = geodesic_errors.shape[0]
    # arr = np.array([((i + 1) / amt_values, x) for (i, x) in zip(range(amt_values), geodesic_errors)])
    # plt.plot(arr[:, 1], arr[:, 0])
    # plt.title("Princeton Benchmark")
    # plt.xlabel("geodesic error")
    # plt.ylabel("% correct correspondences")
    # plt.grid()
    # plt.savefig(f"{file_name}.svg")
    # np.save(f"{file_name}.npy", arr)
    # plt.show()

# test_dataset = load_preprocessed_faust(preprocess_zip, signal_dim=SIGNAL_DIM, kernel_size=KERNEL_SIZE, set_type=2)
#     princeton_benchmark(
#         imcnn=best_model,
#         test_dataset=test_dataset,
#         ref_mesh_path=reference_mesh_path,
#         file_name=f"{log_dir}/best_model_benchmark"
#     )