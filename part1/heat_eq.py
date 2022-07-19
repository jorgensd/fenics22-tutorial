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
#     display_name: Python 3 (DOLFINx complex)
#     language: python
#     name: python3-complex
# ---

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# # The heat equation

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# This notebook will show you how to solve a transient problem using DOLFINx, and highlight differences between `dolfin` and `dolfinx`.
# We start by looking at the structure of DOLFINx:
# -

# Relevant DOLFINx modules
# - `dolfinx.mesh`: Classes and functions related to the computational domain
# - `dolfinx.fem`: Finite element method functionality
# - `dolfinx.io`: Input/Output (read/write) functionality
# - `dolfinx.plot`: Convenience functions for exporting plotting data

# + tags=[]
from dolfinx import mesh, fem, io, plot

# + [markdown] slideshow={"slide_type": "fragment"} tags=[]
# ## Creating a distributed computational domain (mesh)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# To create a simple computational domain in DOLFINx, we use the mesh generation utils in `dolfinx.mesh`. Here we have tools to build rectangles of triangular/quadrilateral and boxes of tetrahedral/hexahedral elements. We start by creating a rectangle spanning [0,0]X[2,3], with 15 and 20 elements in each direction respectively.
# -

from mpi4py import MPI
domain = mesh.create_rectangle(MPI.COMM_WORLD, 
                               [[0.,0.],[2.,3.]],
                               [15, 20], 
                               cell_type=mesh.CellType.quadrilateral)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# As opposed to DOLFIN, we work on simply python structures (nested listes/numpy arrays).
# We also note that we have to send in a communicator. This is because we want the user to be aware of how the mesh is distributed when running in parallel. 
# If we would use the communicator `MPI.COMM_SELF`, each process initialized when running the script would have a mesh local to its process.
# -

# ## Creating a mesh on each process

# + tags=[]
local_domain =  mesh.create_rectangle(MPI.COMM_SELF,
                               [[0.,0.],[2.,3.]],
                               [15, 20], 
                               cell_type=mesh.CellType.quadrilateral)
# -


