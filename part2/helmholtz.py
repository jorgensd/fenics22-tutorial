# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (DOLFINx complex)
#     language: python
#     name: python3-complex
# ---

# + slideshow={"slide_type": "slide"} tags=[]
# The Helmholtz equation

import sys

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
from dolfinx.fem import Function, FunctionSpace
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.io.gmshio import model_to_mesh


if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("This tutorial requires complex number support")
    exit(0)
else:
    print(f"Using {PETSc.ScalarType}.")
# +
try:
    import gmsh
except ModuleNotFoundError:
    print("This demo requires gmsh to be installed")
    sys.exit(0)

# from domain import generate_mesh_wire
from mesh_generation import generate_mesh

k0 = 10 * np.pi
lmbda = 2 * np.pi / k0

model = generate_mesh(lmbda, 2)

mesh, cell_tags, facet_tags = model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=2)
gmsh.finalize()

# +
n = ufl.FacetNormal(mesh)

p = 2

# Definition of function space
element = ufl.FiniteElement("Lagrange", ufl.triangle, p)
V = FunctionSpace(mesh, element)


# Define wave number
DG = FunctionSpace(mesh, ("DG", 0))
k = Function(DG)
k.x.array[:] = k0
k.x.array[cell_tags.values == 1] = 2*k0

x = ufl.SpatialCoordinate(mesh)
ui = ufl.exp(1.0j * k * x[0])
g = ufl.dot(ufl.grad(ui), n) + 1j * k * ui

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Weak Form
ds = ufl.Measure("ds", domain=mesh)
dx = ufl.Measure("dx", domain=mesh)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx \
    - k**2 * ufl.inner(u, v) * dx \
    + 1j * k * ufl.inner(u, v) * ds
L = ufl.inner(g, v) * ds


# Compute solution
uh = Function(V)
uh.name = "u"
problem = LinearProblem(a, L, u=uh, petsc_options={
                        "ksp_type": "preonly", "pc_type": "lu"})
problem.solve()

# XDMF write the solution as a P1 function
with XDMFFile(MPI.COMM_WORLD, "out.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(uh)

# VTX can write higher order function
with VTXWriter(MPI.COMM_WORLD, "out_high_order.bp", uh) as f:
    f.write(0.0)
