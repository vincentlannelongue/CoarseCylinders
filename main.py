import os
import shutil
import numpy as np
from time import time

from mesh_utils import (
    xdmf_to_meshes,
    meshes_to_xdmf,
    interpolate_over_mesh,
    get_surface_mesh_boundaries,
)

from cylinder_utils import generate_cylinder_flow_mesh, create_node_type


MAX_SIZE = 0.04
MIN_SIZE = 0.0075

FIELDS = ["velocity"]
INTERPOLATION_METHODS = ["linear"]


def main():

    root_dir = "/home/admin-vlannelongue/dev/coarse_cylinders"
    xdmf_dir = (
        "/home/admin-vlannelongue/Data/Cylinder/train_set"
    )
    out_dir = os.path.join(root_dir, "cylinder_train_original_size_linear")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(root_dir, exist_ok=True)

    nb_pts = []
    xdmf_files = []
    for xdmf in os.listdir(xdmf_dir):
        if os.path.splitext(xdmf)[1] == ".xdmf":
            xdmf_files.append(os.path.join(xdmf_dir, xdmf))

    done_files = []
    for xdmf in os.listdir(out_dir):
        if os.path.splitext(xdmf)[1] == ".xdmf":
            done_files.append(os.path.split(os.path.splitext(xdmf)[0])[-1])

    for k, xdmf in enumerate(xdmf_files):
        xdmf_filename = os.path.split(os.path.splitext(xdmf)[0])[-1]

        print(f"\n----------- XDMF file {k}/{len(xdmf_files)} - {xdmf_filename}")
        beg_time = time()

        tmp_dir = os.path.join(root_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        coarse_mesh_path = os.path.join(tmp_dir, "coarse_mesh.vtk")
        interpolated_dir = os.path.join(tmp_dir, "interpolated")
        os.makedirs(interpolated_dir, exist_ok=True)

        out_path = os.path.join(out_dir, xdmf_filename)

        if os.path.exists(f"{out_path}.xdmf"):
            print(f"Case {xdmf_filename} already exists, skipping.")
            continue

        # 1 - Extract the vtus from the xdmf archive

        volume_meshes = xdmf_to_meshes(xdmf)  # [:10]
        # TODO: refactor, not good to open all meshes at once, maybe write a generator
        boundaries = get_surface_mesh_boundaries(
            mesh=volume_meshes[0],
            out_path=os.path.join(tmp_dir, "boundaries.vtu"),
        )

        smallest_boundary = min(boundaries, key=len)

        cylinder_boundary = [volume_meshes[0].points[node] for node in smallest_boundary]
        center = np.array(cylinder_boundary).sum(axis=0) / len(cylinder_boundary)
        radius = np.mean([np.linalg.norm(center - n) for n in cylinder_boundary])

        generate_cylinder_flow_mesh(
            xmin=0.0,
            xmax=1.6,
            ymin=0.0,
            ymax=0.41,
            center=center,
            radius=radius,
            min_size=MIN_SIZE,
            max_size=MAX_SIZE,
            out_path=coarse_mesh_path,
            gui=False,
        )

        node_type = create_node_type(
            mesh_file=coarse_mesh_path,
            xmin=0.0,
            xmax=1.6,
            ymin=0.0,
            ymax=0.41,
            center=center,
            radius=radius,
            out_path=os.path.join(tmp_dir, "node_type.vtk"),
        )

        print("done")

        # 4 - Interpolate all vtus onto new mesh
        interpolated_path = os.path.join(tmp_dir, "interpolated")
        os.makedirs(interpolated_path, exist_ok=True)

        interp_meshes = interpolate_over_mesh(
            mesh_file=coarse_mesh_path,
            value_meshes=volume_meshes,
            out_path=interpolated_path,
            out_filename="interp",
            field_names=FIELDS,
            methods=INTERPOLATION_METHODS,
        )

        final_meshes = []
        for i, final_mesh in enumerate(interp_meshes):
            final_mesh.point_data['node_type'] = node_type
            final_meshes.append(final_mesh)

        # 6 - Clean up the temp data

        meshes_to_xdmf(f"{out_path}.xdmf", final_meshes)
        shutil.rmtree(tmp_dir)

        print(f"1 xdmf time: {time()-beg_time}")

    print(nb_pts)


if __name__ == "__main__":
    main()
