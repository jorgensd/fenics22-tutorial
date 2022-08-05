import sys
from xml import dom

try:
    import gmsh
except ModuleNotFoundError:
    print("This tutorial requires gmsh to be installed")
    sys.exit(0)
    
import numpy as np

from dolfinx.graph import create_adjacencylist
from dolfinx.io import distribute_entity_data, XDMFFile
from dolfinx.io.gmshio import extract_geometry, cell_perm_array, ufl_mesh, extract_topology_and_markers
from dolfinx.mesh import CellType, create_mesh, meshtags_from_entities
from mpi4py import MPI


gmsh.initialize(sys.argv)

def generate_mesh(lmbda, order):
    if MPI.COMM_WORLD.rank == 0:

        gmsh.model.add("helmholtz_domain")

        # Recombine tetrahedrons to hexahedrons
        # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        # gmsh.option.setNumber("Mesh.RecombineAll", 2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", lmbda/2)

        # Add scatterers
        c1 = gmsh.model.occ.addCircle(0.0, -1.5*lmbda, 0.0, lmbda/2)
        gmsh.model.occ.addCurveLoop([c1], tag=c1)
        gmsh.model.occ.addPlaneSurface([c1], tag=c1)

        c2 = gmsh.model.occ.addCircle(0.0, 1.5*lmbda, 0.0, lmbda/2)
        gmsh.model.occ.addCurveLoop([c2], tag=c2)
        gmsh.model.occ.addPlaneSurface([c2], tag=c2)

        # Add domain
        r0 = gmsh.model.occ.addRectangle(
            -2*lmbda, -2*lmbda, 0.0, 4*lmbda, 4*lmbda)
        inclusive_rectangle, _ = gmsh.model.occ.fragment(
            [(2, r0)], [(2, c1), (2, c2)])

        gmsh.model.occ.synchronize()

        # Add physical groups
        gmsh.model.addPhysicalGroup(2, [c1, c2], tag=1)
        gmsh.model.addPhysicalGroup(2, [r0], tag=2)

        # Generate mesh
        gmsh.model.mesh.setOrder(order)
        gmsh.model.mesh.generate(2)
        # gmsh.model.mesh.recombine()

        return gmsh.model
