# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# # The Helmholtz equation

import numpy as np

import ufl
from dolfinx.fem import Function, FunctionSpace, assemble_scalar, form
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square
from ufl import dx, grad, inner

from mpi4py import MPI
from petsc4py import PETSc


mesh = create_unit_square(MPI.COMM_WORLD, 128, 128)
n = ufl.FacetNormal(mesh)

p = 2
k0 = 10 * np.pi

# Definition of function space
element = ufl.FiniteElement("Lagrange", ufl.triangle, p)
V = FunctionSpace(mesh, element)


# Incoming wave
# ui.interpolate(incoming_wave)
x = ufl.SpatialCoordinate(mesh)
ui = ufl.exp(1.0j * k0 * x[0])
g = ufl.dot(ufl.grad(ui), n) + 1j * k0 * ui

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Weak Form
ds = ufl.Measure("ds", domain=mesh)
dx = ufl.Measure("dx", domain=mesh)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx \
    - k0**2 * ufl.inner(u, v) * dx \
    + 1j * k0 * ufl.inner(u, v) * ds
L = ufl.inner(g, v) * ds


# Compute solution
uh = Function(V)
uh.name = "u"
problem = LinearProblem(a, L, u=uh, petsc_options={
                        "ksp_type": "preonly", "pc_type": "lu"})
problem.solve()

with XDMFFile(MPI.COMM_WORLD, "out.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(uh)
