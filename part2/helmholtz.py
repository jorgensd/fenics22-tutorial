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

# + [markdown] slideshow={"slide_type": "slide"} tags=[] jp-MarkdownHeadingCollapsed=true
# # The Helmholtz equation
#
# In this tutorial we solve the Helmholtz equation subject to a first order absorbing boundary condition:
#
# $$
# \begin{align*}
# \Delta u + k^2 u &= 0 \qquad \text{in } \Omega,\\
# \nabla u \cdot \mathbf{n} - jku &= g \qquad \text{on } \partial\Omega.
# \end{align*}
# $$ 
# where g is the boundary source term computed as:
#
# $$g = \nabla u_i \cdot \mathbf{n} - jku_i$$
#
# and $u_i$ is the incoming plane wave. 
# + slideshow={"slide_type": "slide"} tags=[]
from mpi4py import MPI
import ufl
from dolfinx.fem import Function, FunctionSpace, petsc
from dolfinx.io import XDMFFile, VTXWriter, gmshio
from dolfinx import plot

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We define some model parameters:

# + slideshow={"slide_type": "fragment"} tags=[]
import numpy as np

k0 = 10 * np.pi
lmbda = 2 * np.pi / k0

# + [markdown] slideshow={"slide_type": "skip"} tags=[]
# This code is only meant to be executed with complex-valued degrees of freedom. To be able to solve such problems, we use the complex build of PETSc.

# + slideshow={"slide_type": "skip"} tags=[]
import sys
from petsc4py import PETSc
if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("This tutorial requires complex number support")
    sys.exit(0)
else:
    print(f"Using {PETSc.ScalarType}.")

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We will use [GMSH](http://gmsh.info/) to generate the computational domain (mesh) for this example. As long as GMSH has been installed (including its Python API), DOLFINx supports direct input of GMSH models (generated on one process). DOLFINx will then in turn distribute the mesh over all processes in the communicator passed to `dolfinx.io.gmshio.model_to_mesh`.

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Interfacing with GMSH

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# The function `generate_mesh` creates a GMSH model on rank 0 of `MPI.COMM_WORLD`.

# + slideshow={"slide_type": "skip"} tags=[]
try:
    import gmsh
except ModuleNotFoundError:
    print("This demo requires gmsh to be installed")
    sys.exit(0)

# + tags=[]
import gmsh
from mesh_generation import generate_mesh
gmsh.initialize()
model = generate_mesh(lmbda, order=2)
mesh, cell_tags, _ = gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0,
                                          gdim=2)
gmsh.finalize()

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Material parameters and boundary conditions

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# In this problem, the wave-number in the different parts of the domain depend on cell markers, inputted through `cell_tags`.
# We use the fact that a Discontinuous Galerkin space of order 0 (cell-wise piecewise constants) has a one-to-one mapping with the cells local to the process.
# -

DG = FunctionSpace(mesh, ("DG", 0))
k = Function(DG)
k.x.array[:] = k0
k.x.array[cell_tags.find(1)] = 2*k0

# +
import pyvista
import matplotlib.pyplot as plt
pyvista.set_jupyter_backend("pythreejs")

# pyvista.start_xvfb(0.5) # Start virtual framebuffer for plotting

topology, cells, geometry = plot.create_vtk_mesh(mesh, 2)
grid = pyvista.UnstructuredGrid(topology, cells, geometry)
grid.cell_data["Marker"] = k.x.array.real
grid2 = grid.tessellate()
grid2.set_active_scalars("Marker")
plotter = pyvista.Plotter()
renderer = plotter.add_mesh(grid2, show_edges=False)
renderer2 = plotter.add_mesh(grid, style="wireframe")#show_edges=True)
plotter.view_xy()

plotter.show()
# img = plotter.screenshot("domains.png", transparent_background=True, window_size=(1000,1000))
# plt.axis("off")
# plt.gcf().set_size_inches(15,15)
# fig = plt.imshow(img)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Next, we define the boundary source term, by using `ufl.SpatialCoordinate`. When using this function, all expressions using this expression will be evaluated at quadrature points.
# -

n = ufl.FacetNormal(mesh)
x = ufl.SpatialCoordinate(mesh)
ui = ufl.exp(-1j * k * x[0])
g = ufl.dot(ufl.grad(ui), n) + 1j * k * ui

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Variational form

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Next, we can define the variational problem, using a 4th order Lagrange space. Note that as we are using complex valued functions, we have to use the appropriate inner product, see [DOLFINx tutorial: Complex numbers](https://jorgensd.github.io/dolfinx-tutorial/chapter1/complex_mode.html) for more information.
#
#
# $$ -\int_\Omega \nabla u \cdot \nabla \bar{v} ~ dx + \int_\Omega k^2 u \,\bar{v}~ dx - j\int_{\partial \Omega} ku  \bar{v} ~ ds = \int_{\partial \Omega} g \, \bar{v}~ ds \qquad \forall v \in \widehat{V}. $$
# -

element = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 4)
V = FunctionSpace(mesh, element)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
ds = ufl.Measure("ds", domain=mesh)
dx = ufl.Measure("dx", domain=mesh)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx \
    - k**2 * ufl.inner(u, v) * dx \
    + 1j * k * ufl.inner(u, v) * ds
L = ufl.inner(g, v) * ds

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Linear solver

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Next, we will solve the problem using a direct solver (LU).

# + tags=[]
problem = petsc.LinearProblem(a, L, petsc_options={
                        "ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
uh.name = "u"

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Visualizing the complex solution
# -
import pyvista
topology, cells, geometry = plot.create_vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cells, geometry)
grid.point_data["Re(u)"] = uh.x.array.real
grid.point_data["Im(u)"] = uh.x.array.imag


# + slideshow={"slide_type": "skip"} tags=[]
pyvista.start_xvfb(0.5) # Start virtual framebuffer for plotting
sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
            position_x=0.1, position_y=0.8, width=0.8, height=0.1)

# + slideshow={"slide_type": "skip"} tags=[]
plotter = pyvista.Plotter()
grid.set_active_scalars("Re(u)")
renderer = plotter.add_mesh(grid, show_edges=False,
                            scalar_bar_args=sargs)
import matplotlib.pyplot as plt
img = plotter.screenshot("Re_u.png",
                         transparent_background=True,
                         window_size=(1000,1000))

# + slideshow={"slide_type": "fragment"} tags=[]
plt.axis("off")
plt.gcf().set_size_inches(10,10)
fig = plt.imshow(img)

# + slideshow={"slide_type": "notes"} tags=[]
plotter_im = pyvista.Plotter()
grid.set_active_scalars("Im(u)")
renderer = plotter_im.add_mesh(grid, show_edges=False,
                            scalar_bar_args=sargs)
img = plotter_im.screenshot("Im_u.png",
                         transparent_background=True,
                         window_size=(1000,1000))

# + slideshow={"slide_type": "slide"} tags=[]
plt.axis("off")
plt.gcf().set_size_inches(15,15)
fig = plt.imshow(img)
# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Saving higher order functions

# + tags=[]
# XDMF write the solution as a P1 function
with XDMFFile(MPI.COMM_WORLD, "out.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(uh)

# VTX can write higher order function
with VTXWriter(MPI.COMM_WORLD, "out_high_order_2.bp", uh) as f:
    f.write(0.0)