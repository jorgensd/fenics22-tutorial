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

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Author: Jørgen S. Dokken

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We start by importing DOLFINx, and check the version and git commit hash

# + tags=[]
import dolfinx
print(f"You have DOLFINx {dolfinx.__version__} installed, " +
      "based on commit \nhttps://github.com/FEniCS/dolfinx/commit/"
      + f"{dolfinx.common.git_commit_hash}")

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Using a 'built-in' mesh

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# No wildcards `*`, as in old DOLFIN, i.e.s

# + [markdown] tags=[]
# ```python
# from dolfin import *
# ```

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We instead import `dolfinx.mesh` as a module

# + slideshow={"slide_type": "fragment"} tags=[]
import dolfinx
from mpi4py import MPI
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Interface to external libraries

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We use external libraries, such as `pyvista` for plotting.import dolfinx.plot
# -

import dolfinx.plot
import pyvista
topology, cells, geometry = dolfinx.plot.create_vtk_mesh(mesh)
grid = pyvista.UnstructuredGrid(topology, cells, geometry)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We add settings for both static and interactive plotting

# + slideshow={"slide_type": "skip"} tags=["hide-cell"]
pyvista.start_xvfb(0.5)
pyvista.set_jupyter_backend("pythreejs")

# + slideshow={"slide_type": "fragment"} tags=[]
plotter = pyvista.Plotter(window_size=(600,600))
renderer = plotter.add_mesh(grid, show_edges=True)

# + slideshow={"slide_type": "skip"} tags=["hide-cell"]
# Settings for presentation mode
plotter.view_xy()
plotter.camera.zoom(1.35)
plotter.export_html("./mesh.html", backend="pythreejs")

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Interactive plot

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We can get interactive plots in notebook by calling.

# + tags=["hide-input"] language="html"
# <iframe src='./mesh.html', scrolling="no", onload='javascript:(function(o){o.style.height=o.contentWindow.document.body.scrollHeight+"px";}(this));'
# style="height:600px;width:100%;border:none;overflow:hidden;">></iframe>
