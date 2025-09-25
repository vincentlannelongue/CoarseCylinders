import enum
import gmsh
import meshio
import numpy as np


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


def generate_cylinder_flow_mesh(
    xmin,
    xmax,
    ymin,
    ymax,
    center,
    radius,
    min_size,
    max_size,
    out_path: str,
    gui: bool = False,
):
    """
    Generates a cylinder flow mesh with boundary layers at top, bottom and cylinder.
    """
    gmsh.initialize()
    gmsh.model.add("cylinder_flow")

    p1 = gmsh.model.occ.addPoint(
        xmin,
        ymin,
        0,
    )
    p2 = gmsh.model.occ.addPoint(xmax, ymin, 0)
    p3 = gmsh.model.occ.addPoint(xmax, ymax, 0)
    p4 = gmsh.model.occ.addPoint(xmin, ymax, 0)

    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p1)
    wall_loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])

    inlet = gmsh.model.addPhysicalGroup(1, [l1])
    gmsh.model.setPhysicalName(1, inlet, "inlet")

    circle = gmsh.model.occ.addCircle(center[0], center[1], 0.0, radius)
    cylinder_loop = gmsh.model.occ.addCurveLoop([circle])

    gmsh.model.occ.addPlaneSurface([wall_loop, cylinder_loop])

    gmsh.model.occ.synchronize()

    # --- Mesh size control ---
    # Distance from hole boundary
    dist_cylinder = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_cylinder, "CurvesList", [circle])
    gmsh.model.mesh.field.setNumber(dist_cylinder, "Sampling", 100)

    # Smooth transition: fine near hole, coarse far away
    thresh_cylinder = gmsh.model.mesh.field.add("Threshold", 4)
    gmsh.model.mesh.field.setNumber(thresh_cylinder, "InField", dist_cylinder)
    gmsh.model.mesh.field.setNumber(
        thresh_cylinder, "SizeMin", min_size
    )  # smallest elements near hole
    gmsh.model.mesh.field.setNumber(
        thresh_cylinder, "SizeMax", max_size
    )  # coarsest elements in far field
    gmsh.model.mesh.field.setNumber(thresh_cylinder, "DistMin", 0.0051)
    gmsh.model.mesh.field.setNumber(thresh_cylinder, "DistMax", 0.15)


    # Distance from walls
    dist_walls = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_walls, "PointsList", [p1, p2, p3, p4])
    gmsh.model.mesh.field.setNumbers(dist_walls, "CurvesList", [l1, l3])
    gmsh.model.mesh.field.setNumber(dist_walls, "Sampling", 100)

    # Smooth transition: fine near hole, coarse far away
    thresh_walls = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh_walls, "InField", dist_walls)
    gmsh.model.mesh.field.setNumber(
        thresh_walls, "SizeMin", min_size * 2
    )  # smallest elements near hole
    gmsh.model.mesh.field.setNumber(
        thresh_walls, "SizeMax", max_size
    )  # coarsest elements in far field
    gmsh.model.mesh.field.setNumber(thresh_walls, "DistMin", 0.01)
    gmsh.model.mesh.field.setNumber(thresh_walls, "DistMax", 0.15)

    # Take the minimum of the two fields
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [thresh_cylinder, thresh_walls])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    # Boundary layer parameters for rectangle
    boundary_layer_rectangle = gmsh.model.mesh.field.add("BoundaryLayer")
    gmsh.model.mesh.field.setNumbers(
        tag=boundary_layer_rectangle,
        option="CurvesList",
        values=[l1, l3],
    )
    gmsh.model.mesh.field.setNumber(
        tag=boundary_layer_rectangle,
        option="Size",
        value=0.0047,
    )
    gmsh.model.mesh.field.setNumber(
        tag=boundary_layer_rectangle, option="Ratio", value=1
    )
    gmsh.model.mesh.field.setNumber(
        tag=boundary_layer_rectangle,
        option="Thickness",
        value=0.0098,
    )

    # Boundary layer parameters for cylinder
    boundary_layer_cylinder = gmsh.model.mesh.field.add("BoundaryLayer")
    gmsh.model.mesh.field.setNumbers(
        tag=boundary_layer_cylinder,
        option="CurvesList",
        values=[circle],
    )
    gmsh.model.mesh.field.setNumber(
        tag=boundary_layer_cylinder,
        option="Size",
        value=0.0025,
    )
    gmsh.model.mesh.field.setNumber(
        tag=boundary_layer_cylinder, option="Ratio", value=1
    )
    gmsh.model.mesh.field.setNumber(
        tag=boundary_layer_cylinder,
        option="Thickness",
        value=0.0051,
    )

    gmsh.model.mesh.field.setAsBoundaryLayer(tag=boundary_layer_rectangle)
    gmsh.model.mesh.field.setAsBoundaryLayer(tag=boundary_layer_cylinder)


    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # Generate mesh:
    gmsh.model.mesh.generate(2)

    # Write mesh data:
    gmsh.write(out_path)
    if gui:
        # Creates  graphical user interface
        gmsh.fltk.run()

        # It finalize the Gmsh API
        gmsh.finalize()


def create_node_type(
    mesh_file, xmin, xmax, ymin, ymax, center, radius, out_path, tol: float = 1e-5
):

    mesh_obj = meshio.read(mesh_file)
    node_type = np.zeros(shape=(len(mesh_obj.points)))
    for i, point in enumerate(mesh_obj.points):
        if point[0] <= xmin + tol:  # inlet
            node_type[i] = NodeType.INFLOW
        elif point[0] >= xmax - tol:  # outlet
            node_type[i] = NodeType.OUTFLOW
        elif np.linalg.norm(center - point[:2]) <= radius + tol:  # cylinder
            node_type[i] = NodeType.WALL_BOUNDARY

        if point[1] <= ymin + tol or point[1] >= ymax - tol:  # walls
            node_type[i] = NodeType.WALL_BOUNDARY

    mesh_obj.point_data = {"node_type": node_type}
    mesh_obj.write(out_path)
    return node_type
