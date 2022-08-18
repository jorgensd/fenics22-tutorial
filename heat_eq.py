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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# # Solving a time-dependent problem

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# This notebook will show you how to solve a transient problem using DOLFINx, and highlight differences between legacy DOLFIN and DOLFINx.
# We start by looking at the structure of DOLFINx:
# -

# Relevant DOLFINx modules:
# - `dolfinx.mesh`: Classes and functions related to the computational domain
# - `dolfinx.fem`: Finite element method functionality
# - `dolfinx.io`: Input/Output (read/write) functionality
# - `dolfinx.plot`: Convenience functions for exporting plotting data
# - `dolfinx.la`: Functions related to linear algebra structures (matrices/vectors)

# + tags=[]
from dolfinx import mesh, fem, io, plot, la

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Creating a distributed computational domain (mesh)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# To create a simple computational domain in DOLFINx, we use the mesh generation utilities in `dolfinx.mesh`. In this module, we have the tools to build rectangles of triangular or quadrilateral elements and boxes of tetrahedral or hexahedral elements. We start by creating a rectangle spanning $[0,0]\times[10,3]$, with 100 and 20 elements in each direction respectively.
# -

from mpi4py import MPI
length, height = 10, 3
Nx, Ny = 80, 60
extent = [[0., 0.], [length, height]]
domain = mesh.create_rectangle(
    MPI.COMM_WORLD, extent, [Nx, Ny], mesh.CellType.quadrilateral)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# In constrast to legacy DOLFIN, we work on simple Python structures (nested listes, numpy arrays, etc).
# We also note that we have to send in an MPI communicator. This is because we want the user to be aware of how the mesh is distributed when running in parallel. 
# If we use the communicator `MPI.COMM_SELF`, each process initialised when running the script would have a version of the full mesh local to its process.

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Creating a mesh on each process

# + tags=[]
local_domain = mesh.create_rectangle(
    MPI.COMM_SELF, extent, [Nx, Ny], mesh.CellType.quadrilateral)
# + [markdown] slideshow={"slide_type": "skip"} tags=[]
# We plot the mesh.

# + slideshow={"slide_type": "skip"} tags=["remove-cell"]
import pyvista
grid = pyvista.UnstructuredGrid(*plot.create_vtk_mesh(local_domain))
plotter = pyvista.Plotter(window_size=(800, 400))
renderer = plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# With Pyvista, we can export the plots to many formats including pngs, interactive notebook plots, and html

# + slideshow={"slide_type": "notes"} tags=["hide-output"]
plotter.view_xy()
plotter.camera.zoom(2)
plotter.export_html("./beam.html", backend="pythreejs")

# + tags=["hide-input"] language="html"
# <iframe src='./beam.html' scrolling="no" onload='javascript:(function(o){o.style.height=o.contentWindow.document.body.scrollHeight+"px";}(this));' style="height:500px;width:100%;border:none;overflow:hidden;"></iframe>  <!--  # noqa, >

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Setting up a variational problem


# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We will solve the heat equation, with a backward Euler time stepping scheme, ie
# -

# $$
# \begin{align*}
# \frac{u_{n+1}-u_n}{\Delta t} - \nabla \cdot (\mu  \nabla u_{n+1}) &= f(x,t_{n+1}) && \text{in } \Omega,\\
# u &= u_D(x,t_{n+1}) &&\text{on } \partial\Omega_\text{D},\\
# \mu\frac{\partial u_{n+1}}{\partial n} &=0 &&\text{on } \partial\Omega_\text{N},
# \end{align*}
# $$ 
# with $u_D = y\cos(0.25t)$, $f=0$. For this example, we take $\Omega$ to be rectangle defined above, $\Omega_\text{D}$ if the left-hand edge of the rectangle, and $\Omega_\text{N}$ is the remaining three edges of the rectangle.

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We start by defining the function space, the corresponding test and trial functions, as well as material and temporal parameters. Note that we use explicit imports from UFL to create the test and trial functions, to avoid confusion as to where they originate from. DOLFINx and UFL support both real and complex valued functions. However, to be able to use the PETSc linear algebra backend, which only supports a single floating type at compilation, we need to use appropriate scalar types in our variational form. This ensures that we generate consistent matrices and vectors.
# -

from ufl import TestFunction, TrialFunction, dx, grad, inner, system
V = fem.FunctionSpace(domain, ("Lagrange", 1))
u = TrialFunction(V)
v = TestFunction(V)
un = fem.Function(V)
f = fem.Constant(domain, 0.0)
mu = fem.Constant(domain, 2.3)
dt = fem.Constant(domain, 0.05)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# The variational form can be written in UFL syntax, as done in legacy DOLFIN:

# + slideshow={"slide_type": "fragment"} tags=[]
F = inner(u - un, v) * dx + dt * mu * inner(grad(u), grad(v)) * dx
F -= dt * inner(f, v) * dx
(a, L) = system(F)

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Creating Dirichlet boundary conditions

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# ### Creating a time dependent boundary condition

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# There are many ways of creating boundary conditions. In this example, we will create function $u_\text{D}(x,t)$ dependent on both space and time. To do this, we define a function that takes a 2-dimensional array `x`.  Each column of `x` corresponds to an input coordinate $(x,y,z)$ and this function operates directly on the columns of `x`.

# +
import numpy as np

def uD_function(t):
    return lambda x: x[1] * np.cos(0.25 * t)

uD = fem.Function(V)
t = 0
uD.interpolate(uD_function(t))


# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# To give the user freedom to set boundary conditions on single degrees of freedom, the function `dolfinx.fem.dirichletbc` takes in the list of degrees of freedom (DOFs) as input. The DOFs on the boundary can be obtained in many ways: DOLFINx supplies a few convenience functions, such as `dolfinx.fem.locate_dofs_topological` and `dolfinx.fem.locate_dofs_geometrical`.
# Locating dofs topologically is generally advised, as certain finite elements have DOFs that do not have a geometrical coordinates associated with them (eg Nédélec and Raviart--Thomas). DOLFINx also has convenicence functions to obtain a list of all boundary facets.

# + slideshow={"slide_type": "fragment"} tags=[]
def dirichlet_facets(x):
    return np.isclose(x[0], length)

tdim = domain.topology.dim
bc_facets = mesh.locate_entities_boundary(
    domain, tdim - 1, dirichlet_facets)

# + slideshow={"slide_type": "fragment"} tags=[]
bndry_dofs = fem.locate_dofs_topological(V, tdim - 1, bc_facets)

# + slideshow={"slide_type": "fragment"} tags=[]
bcs = [fem.dirichletbc(uD, bndry_dofs)]

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Setting up a time dependent solver

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# As the left hand side of our problem (the matrix) is time independent, we would like avoid re-assembling it at every time step. DOLFINx gives the user more control over assembly so that this can be done. We assemble the matrix once outside the temporal loop.
# -
compiled_a = fem.form(a)
A = fem.petsc.assemble_matrix(compiled_a, bcs=bcs)
A.assemble()

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Next, we can generate the integration kernel for the right hand side (RHS), and create the RHS vector `b` that we will assemble into at each time step.

# + slideshow={"slide_type": "fragment"} tags=[]
compiled_L = fem.form(L)
b = fem.Function(V)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We next create the PETSc KSP (Krylov subspace method) solver, and set it to solve using an [algebraic multigrid method](https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html).

# + slideshow={"slide_type": "fragment"} tags=[]
from petsc4py import PETSc
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.CG)
pc = solver.getPC()
pc.setType(PETSc.PC.Type.HYPRE)
pc.setHYPREType("boomeramg")

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Plotting a time dependent problem
#

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# As we are solving a time dependent problem, we would like to create a time dependent animation of the solution. 
# We do this by using [pyvista](https://docs.pyvista.org/), which uses VTK structures for plotting.
# In DOLFINx, we have the convenience function `dolfinx.plot.create_vtk_mesh` that can create meshes compatible with VTK formatting, based on meshes of (discontinuous) Lagrange function spaces.

# + tags=["hide-cell"]
import pyvista
import matplotlib.pyplot as plt
pyvista.start_xvfb(0.5)  # Start virtual framebuffer for plotting
plotter = pyvista.Plotter()
plotter.open_gif("u_time.gif")

# + slideshow={"slide_type": "fragment"} tags=[]
topology, cells, geometry = plot.create_vtk_mesh(V)
uh = fem.Function(V)
grid = pyvista.UnstructuredGrid(topology, cells, geometry)
grid.point_data["uh"] = uh.x.array

# + slideshow={"slide_type": "skip"} tags=[]
viridis = plt.cm.get_cmap("viridis", 25)
sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
             position_x=0.1, position_y=0.8, width=0.8, height=0.1)

# + slideshow={"slide_type": "fragment"} tags=[]
renderer = plotter.add_mesh(grid, show_edges=True, lighting=False,
                            cmap=viridis, scalar_bar_args=sargs,
                            clim=[0, height])

# + slideshow={"slide_type": "skip"} tags=[]
plotter.view_xy()
plotter.camera.zoom(1.3)

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Solving a time dependent problem

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We are now ready to solve the time dependent problem. At each time step, we need to:
# 1. Update the time dependent boundary condition and source
# 2. Reassemble the right hand side vector `b`
# 3. Apply boundary conditions to `b`
# 4. Solve linear problem `Au = b`
# 5. Update previous time step, `un = u`
# -

T = 2 * np.pi
while t < T:
    # Update boundary condition
    t += dt.value
    uD.interpolate(uD_function(t))

    # Assemble RHS
    b.x.array[:] = 0
    fem.petsc.assemble_vector(b.vector, compiled_L)

    # Apply boundary condition
    fem.petsc.apply_lifting(b.vector, [compiled_a], [bcs])
    b.x.scatter_reverse(la.ScatterMode.add)
    fem.petsc.set_bc(b.vector, bcs)

    # Solve linear problem
    solver.solve(b.vector, uh.vector)
    uh.x.scatter_forward()

    # Update un
    un.x.array[:] = uh.x.array

    # Update plotter
    plotter.update_scalars(uh.x.array, render=False)
    plotter.write_frame()
# + slideshow={"slide_type": "skip"} tags=[]
plotter.close()

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# <img src="./u_time.gif" alt="gif" class="bg-primary mb-1" width="800px">