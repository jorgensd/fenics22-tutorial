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

# + [markdown] slideshow={"slide_type": "slide"} tags=[] jp-MarkdownHeadingCollapsed=true tags=[] slideshow={"slide_type": "slide"}
# # The Helmholtz equation
# + [markdown] slideshow={"slide_type": "skip"} tags=[]
# Author: Igor A. Baratta
# -

# In this tutorial, we will learn:
#
#  - How to solve PDEs with complex-valued fields,
#  - How to import and use high-order meshes from Gmsh,
#  - How to use high order discretizations,
#  - How to use UFL expressions.

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Problem statement
# We will solve the Helmholtz equation subject to a first order absorbing boundary condition:
#
# $$
# \begin{align*}
# \Delta u + k^2 u &= 0 \qquad \text{in } \Omega,\\
# \nabla u \cdot \mathbf{n} - jku &= g \qquad \text{on } \partial\Omega.
# \end{align*}
# $$
#
# where $k$ is a pieciwise constant wavenumber, and $g$ is the boundary source term computed as:
#
# $$g = \nabla u_i \cdot \mathbf{n} - jku_i$$
#
# and $u_i$ is the incoming plane wave. 

# + slideshow={"slide_type": "skip"} tags=[]
import numpy as np
from mpi4py import MPI

import dolfinx
import ufl

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

# + [markdown] slideshow={"slide_type": "skip"} tags=[]
# ## Defining model parameters

# + slideshow={"slide_type": "skip"} tags=[]
# wavenumber in free space (air)
k0 = 10 * np.pi

# Corresponding wavelength
lmbda = 2 * np.pi / k0

# Polynomial degree
degree = 6

# Mesh order
mesh_order = 2

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Interfacing with GMSH

# + [markdown] slideshow={"slide_type": "skip"} tags=[]
# We will use [Gmsh](http://gmsh.info/) to generate the computational domain (mesh) for this example. As long as Gmsh has been installed (including its Python API), DOLFINx supports direct input of Gmsh models (generated on one process). DOLFINx will then in turn distribute the mesh over all processes in the communicator passed to `dolfinx.io.gmshio.model_to_mesh`.
#
# The function `generate_mesh` creates a Gmsh model on rank 0 of `MPI.COMM_WORLD`.The function `generate_mesh` creates a Gmsh model on rank 0 of `MPI.COMM_WORLD`.

# + tags=[]
from dolfinx.io import gmshio
from mesh_generation import generate_mesh

# MPI communicator
comm = MPI.COMM_WORLD

file_name = "domain.msh"
generate_mesh(file_name, lmbda, order=mesh_order)
mesh, cell_tags, _ = gmshio.read_from_msh(file_name, comm, 
                                          rank = 0, gdim = 2)

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Material parameters

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# In this problem, the wave number in the different parts of the domain depends on cell markers, inputted through `cell_tags`.
# We use the fact that a Discontinuous Lagrange space of order 0 (cell-wise piecewise constants) has a one-to-one mapping with the cells local to the process.

# + tags=[]
W = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
k = dolfinx.fem.Function(W)
k.x.array[:] = k0
k.x.array[cell_tags.find(1)] = 3*k0

# + slideshow={"slide_type": "skip"} tags=[]
import pyvista
import matplotlib.pyplot as plt
from dolfinx.plot import create_vtk_mesh

# Start virtual framebuffer for plotting
pyvista.start_xvfb(0.5)
pyvista.set_jupyter_backend("pythreejs")
pyvista.set_plot_theme("paraview")

sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
            position_x=0.1, position_y=0.8, width=0.8, height=0.1)

def export_function(grid, name, show_mesh=False):

    grid.set_active_scalars(name)
    plotter = pyvista.Plotter(window_size=(800,800))
    renderer = plotter.add_mesh(grid, show_edges=False, scalar_bar_args=sargs)
    if show_mesh:
        V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))
        grid_mesh = pyvista.UnstructuredGrid(*create_vtk_mesh(V))
        renderer = plotter.add_mesh(grid_mesh, style="wireframe", line_width=0.1, color="k")
        plotter.view_xy()
    plotter.view_xy()
    plotter.camera.zoom(1.3)
    plotter.export_html(f"./{name}.html", backend="pythreejs")


# + tags=["ignore-output", "hide-input"]
grid = pyvista.UnstructuredGrid(*create_vtk_mesh(mesh))
grid.cell_data["wavenumber"] = k.x.array.real
export_function(grid, "wavenumber", show_mesh=True)


# + tags=["hide-input"] language="html"
# <iframe src='./wavenumber.html', scrolling="no", onload='javascript:(function(o){o.style.height=o.contentWindow.document.body.scrollHeight+"px";}(this));'
# style="height:800px;width:100%;border:none;overflow:hidden;">></iframe>

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Boundary source term
#
# $$g = \nabla u_{inc} \cdot \mathbf{n} - \mathrm{j}ku_{inc}$$

# + [markdown] slideshow={"slide_type": "skip"} tags=[]
#
# Next, we define the boundary source term, by using `ufl.SpatialCoordinate`. When using this function, all quantities using this expression will be evaluated at quadrature points.

# + tags=[]
n = ufl.FacetNormal(mesh)
x = ufl.SpatialCoordinate(mesh)
uinc = ufl.exp(-1j * k * x[0])
g = ufl.dot(ufl.grad(uinc), n) + 1j * k * uinc

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Variational form

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Next, we can define the variational problem, using a 4th order Lagrange space. Note that as we are using complex valued functions, we have to use the appropriate inner product, see [DOLFINx tutorial: Complex numbers](https://jorgensd.github.io/dolfinx-tutorial/chapter1/complex_mode.html) for more information.
# -

# $$ -\int_\Omega \nabla u \cdot \nabla \bar{v} ~ dx + \int_\Omega k^2 u \,\bar{v}~ dx - j\int_{\partial \Omega} ku  \bar{v} ~ ds = \int_{\partial \Omega} g \, \bar{v}~ ds \qquad \forall v \in \widehat{V}. $$

# + slideshow={"slide_type": "fragment"} tags=[]
element = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree)
V = dolfinx.fem.FunctionSpace(mesh, element)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# + tags=[] slideshow={"slide_type": "fragment"}
a = - ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
    + k**2 * ufl.inner(u, v) * ufl.dx \
    - 1j * k * ufl.inner(u, v) * ufl.ds
L = ufl.inner(g, v) * ufl.ds

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Linear solver

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Next, we will solve the problem using a direct solver (LU).

# + tags=[]
opt = {"ksp_type": "preonly", "pc_type": "lu"}
problem = dolfinx.fem.petsc.LinearProblem(a, L, petsc_options=opt)
uh = problem.solve()
uh.name = "u"

# + [markdown] slideshow={"slide_type": "skip"} tags=[]
# ## Postprocessing
#
# ### Visualising using PyVista
# + tags=[] slideshow={"slide_type": "skip"}
topology, cells, geometry = create_vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cells, geometry)
grid.point_data["Abs(u)"] = np.abs(uh.x.array)


# + tags=["remove-input"]
export_function(grid, "Abs(u)")

# + slideshow={"slide_type": "skip"} tags=["remove-cell"]
# Open a gif
plotter = pyvista.Plotter()
plotter.open_gif("wave.gif")

boring_cmap = plt.cm.get_cmap("coolwarm", 25)

pts = grid.points.copy()
warped = grid.warp_by_scalar()
renderer = plotter.add_mesh(warped, show_edges=False, clim=[-2, 2.5],lighting=False, cmap=boring_cmap)

nframe = 27
for phase in np.linspace(0, 2 * np.pi, nframe + 1)[:nframe]:
    data = uh.x.array * np.exp(1j*phase)
    plotter.update_scalars(data.real, render=False)
    warped = grid.warp_by_scalar(factor=0.1)
    plotter.update_coordinates(warped.points.copy(), render=False)
    
    # Write a frame. This triggers a render.
    plotter.write_frame()
plotter.close()

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# <img src="./wave.gif" alt="gif" class="bg-primary mb-1" width="800px">

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ### Post-processing with Paraview

# + tags=[]
from dolfinx.io import XDMFFile, VTXWriter
u_abs = dolfinx.fem.Function(V, dtype=np.float64)
u_abs.x.array[:] = np.abs(uh.x.array)

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ### XDMFFile
# -

# XDMF writes data to mesh nodes
with XDMFFile(comm, "out.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(u_abs)

# <img src="./img/xdmf.png" alt="xdmf" class="bg-primary mb-1" width="500px">

# + [markdown] slideshow={"slide_type": "slide"} tags=["ignore-output"]
# ### VTXWriter
# -

# VTX can write higher order functions
with VTXWriter(comm, "out_high_order.bp", [u_abs]) as f:
    f.write(0.0)

# <img src="./img/vtx.png" alt="vtx" class="bg-primary mb-1" width="500px">


