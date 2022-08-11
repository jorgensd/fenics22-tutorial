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
# # Introduction to DOLFINx
# We start by importing DOLFINx, and check the version and git commit hash

# + tags=[]
import dolfinx
print(f"You have DOLFINx {dolfinx.__version__} installed, " +
      "based on commit \nhttps://github.com/FEniCS/dolfinx/commit/"
      + f"{dolfinx.common.git_commit_hash}")

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Using a 'built-in' mesh
# No wildcards `*`, as in old DOLFIN, i.e.
# ```python
# from dolfin import *
# ```

# + [markdown] slideshow={"slide_type": "fragment"} tags=[]
# We instead import `dolfinx.mesh` as a module

# + tags=[]
import dolfinx
from mpi4py import MPI
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Interface to external libraries
# We use external libraries, such as `pyvista` for plotting.import dolfinx.plot
# -

import dolfinx.plot
import pyvista
topology, cells, geometry = dolfinx.plot.create_vtk_mesh(mesh)
grid = pyvista.UnstructuredGrid(topology, cells, geometry)

# We add settings for both static and interactive plotting

# + slideshow={"slide_type": "skip"} tags=[]
pyvista.start_xvfb(0.5)
pyvista.set_jupyter_backend("pythreejs")
# -

plotter = pyvista.Plotter()
renderer = plotter.add_mesh(grid, show_edges=True)

# + slideshow={"slide_type": "skip"} tags=["hide-cell"]
# Settings for presentation mode
plotter.view_xy()
plotter.camera.zoom(1.35)
# -

img = plotter.screenshot("fundamentals_mesh.png",
                         transparent_background=True,
                         window_size=(800,800))

# + [markdown] slideshow={"slide_type": "slide"} tags=["hide-cell"]
# ## Plot mesh using matplotlib
# -

# In the hidden cell you can expand the settings to get a static plot using matplotlib. 

# + tags=["hide-cell"]
import matplotlib.pyplot as plt
plt.axis("off")
plt.gcf().set_size_inches(7,7)
fig = plt.imshow(img)

# + [markdown] slideshow={"slide_type": "skip"} tags=[]
# ## Interactive plot
# We can get interactive plots in notebook by calling.

# + slideshow={"slide_type": "skip"} tags=[]
plotter.show()
