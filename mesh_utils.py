import itertools
import meshio
import networkx
import numpy as np
import os
from scipy.interpolate import griddata
import shutil
from tqdm import tqdm
from typing import List, Union


def xdmf_to_meshes(xdmf_file_path: str, verbose: bool = False) -> List[meshio.Mesh]:
    """
    Returns meshio mesh objects for every timestep in an XDMF archive file.
    """
    reader = meshio.xdmf.TimeSeriesReader(xdmf_file_path)
    points, cells = reader.read_points_cells()
    meshes = []
    for i in tqdm(range(reader.num_steps), desc='Extracting meshes from XDMF file', disable=not verbose):
        time, point_data, cell_data, _ = reader.read_data(i)
        mesh = meshio.Mesh(points, cells, point_data=point_data, cell_data=cell_data)
        meshes.append(mesh)
    return meshes


def meshes_to_xdmf(filename: str, meshes: List[meshio.Mesh], timestep=1, verbose: bool = False) -> None:
    """
    Writes a time series of meshes (same points and cells) into XDMF/HDF5 format.
    The function will write two files: 'filename.xdmf' and 'filename.h5'.

    filename: chosen name for the archive files.
    meshes: List of meshes to compress, they need to share their cells and points.
    timestep: Timestep betwwen two frames.
    """
    # TODO: allowing for a timeframe list could be useful

    points = meshes[0].points
    cells = meshes[0].cells
    cells = [cells[2]]  # ONLY FOR 2D

    filename = os.path.splitext(filename)[0]
    h5_filename = f"{filename}.h5"
    xdmf_filename = f"{filename}.xdmf"

    # Open the TimeSeriesWriter for HDF5
    with meshio.xdmf.TimeSeriesWriter(xdmf_filename) as writer:
        # Write the mesh (points and cells) once
        writer.write_points_cells(points, cells)

        # Loop through time steps and write data
        t = 0
        for mesh in tqdm(meshes, desc='Compressing mesh into XDMF files', disable=not verbose):
            point_data = mesh.point_data
            cell_data = mesh.cell_data
            writer.write_data(t, point_data=point_data, cell_data=cell_data)
            t += timestep

    # The H5 archive is systematically created in cwd with the original meshio library
    if os.path.exists(os.path.join(os.getcwd(), os.path.split(h5_filename)[1])):
        shutil.move(src=os.path.join(os.getcwd(), os.path.split(h5_filename)[1]), dst=h5_filename)
    print(f"Time series written to {xdmf_filename} and {h5_filename}")


def interpolate_over_mesh(mesh_file: Union[str, meshio.Mesh],
                          field_names: Union[str, List[str]],
                          value_meshes: Union[List[str], List[meshio.Mesh]],
                          out_path: str = None,
                          out_filename: str = None,
                          new_field_names: List[str] = None,
                          scaling_factor: float = 1,
                          methods: Union[str, List[str]] = 'nearest',
                          fill_value: float = 0.0,
                          verbose: bool = True) -> List[meshio.Mesh]:
    """
    Gather values from a vtu and interpolate these values on a defined mesh.

    Args:
        mesh_file : File (vtu or vtk) containing the mesh we interpolate values on.
        value_meshes : List of paths to meshes, or meshes with the values we want to extract.
        field_name : Field name for the values we consider from the mesh.
        new_field_name : Optional, name for the interpolated field if changing.
        out_path : Optionnal, path to the output directory. If None, the meshes aren't saved.
        out_filename : Filename to iterate over for output files.
        scaling_factor : Factor to scale the value mesh coordinates and the values.
        methods : Interpolation method, between 'nearest' and 'linear', for each field.
    Returns:
        List of new interpolated meshes.
    """
    # Handle the possibility that methods is a single method
    if type(methods) is str:
        methods = [methods]*len(field_names)

    if type(mesh_file) is str:
        mesh_file = meshio.read(mesh_file)
    # We open all the meshes at once, might be a problem
    if type(value_meshes[0]) is str:
        value_meshes = [meshio.read(value_mesh_file) for value_mesh_file in value_meshes]
    if type(field_names) is str:
        field_names = [field_names]
    if new_field_names is None:
        new_field_names = field_names
    if (out_path and not out_filename) or (not out_path and out_filename):
        raise ValueError("To save the interpolated meshes, need both 'out_path' and 'out_filename' arguments.")

    out_meshes = []
    for i, value_mesh in enumerate(tqdm(value_meshes, ncols=50, desc='Interpolating over mesh', disable=not verbose)):
        value_points = value_mesh.points*scaling_factor

        out_mesh = mesh_file.copy()
        out_mesh.point_data = dict()

        for field, new_field, method in zip(field_names, new_field_names, methods):
            value_data = value_mesh.point_data[field]*scaling_factor

            interpolation = griddata(points=value_points,
                                     values=value_data,
                                     xi=out_mesh.points[:, :2],
                                     method=method,
                                     fill_value=fill_value)

            out_mesh.point_data[new_field] = interpolation

        if out_path:
            out_mesh.write(os.path.join(out_path, f"{out_filename}_{i:05d}.vtu"))  # , binary=False)
        out_meshes.append(out_mesh)
    return out_meshes


def get_edges_from_cells(cells: np.ndarray, dimension: int) -> List[List[int]]:
    """
    Get edges from cells based on the dimension.
    Args:
        cells (np.ndarray): Array of cells of shape (N, dimension+1), each row is a cell defined by its vertices.
        dimension (int): Dimension of the mesh elements (2 for triangle, 3 for tetrahedras).
    Returns:
        List[List[int]]: List of edges (or triangles) extracted from the cells.
    """
    edges = []
    for cell in cells:
        for el in itertools.combinations(range(dimension + 1), dimension):
            edge = sorted([cell[el[i]] for i in range(dimension)])
            edges.append(edge)
    return edges


def find_closed_loops(edges):
    """
    Find all closed loops in a set of edges.

    Parameters
    ----------
    edges : ndarray (N, 2)
        Array of edges, each row is a pair of node indices.

    Returns
    -------
    loops : list of lists
        Each loop is a list of node indices (start and end node will be the same).
    """
    G = networkx.Graph()
    G.add_edges_from(edges)

    # Find all simple cycles
    loops = list(networkx.cycle_basis(G))

    # Optionally, you can add the start node at the end to make loops explicit
    loops = [loop + [loop[0]] for loop in loops]

    return loops


def get_boundary_edges(cells: np.ndarray, dimension: int = 2) -> List[List[int]]:
    """
    Get the boundary cells (edges in 2D, triangles in 3D) of a graph.
    Args:
        cells (np.ndarray): Array of cells of shape (N, dimension+1), each row is a cell defined by its vertices.
        dimension (int): Dimension of the mesh elements (2 for triangle, 3 for tetrahedras).
    Returns:
        List[List[int]]: List of boundary edges (or triangles).
    """
    edges = get_edges_from_cells(cells, dimension)

    if dimension == 2:
        str_edges = [f"{e[0]}, {e[1]}" for e in edges]
    elif dimension == 3:
        str_edges = [f"{e[0]}, {e[1]}, {e[2]}" for e in edges]

    # Count the number of occurence of each edge
    edge_count = dict()
    for edge in str_edges:
        if edge in edge_count:
            edge_count[edge] += 1
        else:
            edge_count[edge] = 1

    # Get only the edges that occur once -> they are the boundary edges
    boundary_str_edges = [edge for edge, count in edge_count.items() if count == 1]
    boundary_edges = [
        [int(e.split(",")[i]) for i in range(dimension)] for e in boundary_str_edges
    ]
    return boundary_edges


def get_surface_mesh_boundaries(
    mesh: Union[str, meshio.Mesh],
    out_path: str = None,
    dimension: int = 2,
    group_by_loops: bool = False,
) -> List[List[int]]:
    """
    Generates the a binary mask indicating if a node is on the mesh wall boundary (line for 2D meshes, surface for 3D).
    Args:
        mesh: meshio Mesh object, or path to a mesh file.
        out_path: Optionnal, path to save the mesh with the wall_mask as point data.
        dimension: Dimension of the mesh; 2 for triangular mesh, 3 for tetrahedric.

    Returns:
        List[List[int]]: A list of lists of nodes constituting the boundaries.
    """
    if type(mesh) is str:
        mesh = meshio.read(mesh)
    if dimension == 2:
        cell_type = "triangle"
    elif dimension == 3:
        cell_type = "tetra"

    # Get all cells of dim 'dimension - 1' (triangles for 3D, edges for 2D)
    cells = mesh.cells_dict[cell_type]
    boundary_edges = get_boundary_edges(cells, dimension)

    grouped_boundary_nodes = find_closed_loops(boundary_edges)
    # else:  # No node repetition here
    #     grouped_boundary_nodes = find_connected_components(boundary_edges)

    wall_mask = np.zeros(shape=(len(mesh.points)))
    for i, boundary_nodes in enumerate(grouped_boundary_nodes):
        for indx in boundary_nodes:
            wall_mask[indx] = 1
    mesh.point_data["bound"] = wall_mask
    if out_path:
        mesh.write(out_path)

    return grouped_boundary_nodes
