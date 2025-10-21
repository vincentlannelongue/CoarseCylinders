import h5py
import json
import meshio
import os

from mesh_utils import meshes_to_xdmf


def convert_h5_to_xdmf(h5_file, xdmf_directory, meta_path):
    """
    Convert an HDF5 file to XDMF format.
    Args:
        h5_file (str): Path to the input HDF5 file.
        xdmf_directory (str): Path to the output XDMF directory.
    """
    file_handle = h5py.File(h5_file, "r")
    datasets_index = list(file_handle.keys())

    trajectories = []

    with open(meta_path, "r") as fp:
        meta = json.load(fp)

    for traj_number in datasets_index:
        features = file_handle[traj_number]
        traj = {}

        for key, field in meta["features"].items():
            data = features[key][()].astype(field["dtype"])
            data = data.reshape(field["shape"])
            traj[key] = data
        trajectories.append(traj)

    for i, traj in enumerate(trajectories):
        traj_meshes = []
        for t in range(len(traj["mesh_pos"])):
            mesh = meshio.Mesh(
                points=traj["mesh_pos"][t],
                cells={"triangle": traj["cells"][t]},
                point_data={"velocity": traj["velocity"][t],
                            "pressure": traj["pressure"][t],
                            "node_type": traj["node_type"][t]},

            )
            traj_meshes.append(mesh)

        meshes_to_xdmf(
            filename=os.path.join(xdmf_directory, f"trajectory_{i}"),
            meshes=traj_meshes,
            timestep=1,
            verbose=True,
        )

    print("done")


data_path = "/home/admin-vlannelongue/Data/Cylinder/DeepmindData"
h5_file = os.path.join(data_path, "train.h5")
outdir = os.path.join(data_path, "train_set")
os.makedirs(outdir)

convert_h5_to_xdmf(
    h5_file,
    outdir,
    "cylinder_meta.json",
)
