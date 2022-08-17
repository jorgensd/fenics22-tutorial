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
# # Stokes equation

# + [markdown] slideshow={"slide_type": "skip"} tags=[]
# Authors: J.S. Dokken, M.W. Scroggs, S. Roggendorf

# + [markdown] tags=[]
# $$\begin{align}
# -\Delta \mathbf{u} + \nabla p &= \mathbf{f}(x,y) \quad \text{in } \Omega\\
# \nabla \cdot \mathbf{u} &= 0 \qquad\quad \text{in } \Omega\\
# \mathbf{u} &= \mathbf{0}\qquad\quad \text{on } \partial\Omega
# \end{align}
# $$
# In this tutorial you will learn how to:

# + [markdown] slideshow={"slide_type": "fragment"} tags=[]
# - Create manufactured solutions with UFL
# - Use block-preconditioners
# - Use custom finite elements

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Imports

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We start by importing most of the modules we will use in this tutorial.

# + tags=[]
from dolfinx import fem, mesh

from mpi4py import MPI
from petsc4py import PETSc

import basix, basix.ufl_wrapper
import matplotlib.pylab as plt
import numpy as np

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# However, we pay special attention to `UFL`, the unified form language package, which is used to represent variational forms.
# As we will dependend on alot of functions from this package, we 
# import the components that we will use in this code explicitly

# + slideshow={"slide_type": "fragment"} tags=[]
from ufl import (VectorElement, EnrichedElement, FiniteElement,
                 SpatialCoordinate, TrialFunction, TestFunction,
                 as_vector, cos, sin, inner, div, grad, dx, pi)


# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Defining a manufactured solution

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We will use a known analytical solution to the Stokes equations in this tutorial. We define the exact velocity and pressure as the following:

# +
def u_ex(x):
    sinx = sin(pi * x[0]) 
    siny = sin(pi * x[1])
    cosx = cos(pi * x[0])
    cosy = cos(pi * x[1])
    c_factor = 2 * pi * sinx * siny
    return c_factor * as_vector((cosy * sinx, - cosx * siny))

def p_ex(x):
    return sin(2 * pi * x[0]) * sin(2 * pi * x[1])


# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Here the input to each function is the coordinates (x,y) of the problem. These will in turn be defined by using `x = ufl.SpatialCoordinate(domain)`.
#
# We use the strong formulation of the PDE to compute the source function $\mathbf{f}$ using UFL operators

# + slideshow={"slide_type": "fragment"} tags=[]
def source(x):
    u, p = u_ex(x), p_ex(x)
    return - div(grad(u)) + grad(p)


# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Defining the variational form

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We will solve the PDE by creating a set of variational forms, one for each component of the problem
# -

# $$\begin{align}
# A w &= b,\\
# \begin{pmatrix}
# A_{\mathbf{u},\mathbf{u}} & A_{\mathbf{u},p} \\
# A_{p,\mathbf{u}} & 0
# \end{pmatrix}
# \begin{pmatrix} u\\ p \end{pmatrix}
# &= \begin{pmatrix}\mathbf{f}\\ 0 \end{pmatrix}
# \end{align}$$

# + slideshow={"slide_type": "fragment"} tags=[]
def create_bilinear_form(V, Q):
    u, p = TrialFunction(V), TrialFunction(Q)
    v, q = TestFunction(V), TestFunction(Q)
    a_uu = inner(grad(u), grad(v)) * dx
    a_up = inner(p, div(v)) * dx
    a_pu = inner(div(u), q) * dx
    return fem.form([[a_uu, a_up], [a_pu, None]])


# + slideshow={"slide_type": "fragment"} tags=[]
def create_linear_form(V, Q):
    v, q = TestFunction(V), TestFunction(Q)
    domain = V.mesh
    x = SpatialCoordinate(domain)
    f = source(x)
    return fem.form([inner(f, v) * dx,
                     inner(fem.Constant(domain, 0.), q) * dx])


# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Boundary conditions
# -

def create_velocity_bc(V):
    domain = V.mesh
    g = fem.Constant(domain, [0.,0.])
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    bdry_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, tdim - 1, bdry_facets)
    return [fem.dirichletbc(g, dofs, V)]


# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# In the problem description above, we have only added a boundary condition for the velocity.
# This means that the problem is singular, i.e. the pressure is only determined up to a constant. We therefore create a PETSc nullspace operator for the pressure.

# + slideshow={"slide_type": "fragment"} tags=[]
def create_nullspace(rhs_form):
    null_vec = fem.petsc.create_vector_nest(rhs_form)
    null_vecs = null_vec.getNestSubVecs()
    null_vecs[0].set(0.0)
    null_vecs[1].set(1.0)
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    return nsp


# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Create a block preconditioner

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We create a nested matrix `P` to use as the preconditioner.
# The top-left block of `P` is the top-left block of `A`. 
# The bottom-right diagonal entry is a mass matrix.
# -

def create_preconditioner(Q, a, bcs):
    p, q = TrialFunction(Q), TestFunction(Q)
    a_p11 = fem.form(inner(p, q) * dx)
    a_p = fem.form([[a[0][0], None],
                    [None, a_p11]])
    P = fem.petsc.assemble_matrix_nest(a_p, bcs)
    P.assemble()
    return P


# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Assemble nest system
# -

def assemble_system(lhs_form, rhs_form, bcs):
    A = fem.petsc.assemble_matrix_nest(lhs_form, bcs=bcs)
    A.assemble()
    
    b = fem.petsc.assemble_vector_nest(rhs_form)
    fem.petsc.apply_lifting_nest(b, lhs_form, bcs=bcs)
    for b_sub in b.getNestSubVecs():
        b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD,
                          mode=PETSc.ScatterMode.REVERSE)
    spaces = fem.extract_function_spaces(rhs_form)
    bcs0 = fem.bcs_by_block(spaces, bcs)
    fem.petsc.set_bc_nest(b, bcs0)
    return A, b


# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## PETSc Krylov Subspace solver

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# In legacy DOLFIN, convenience functions were provided to interact with linear algebra packages such as PETSc.
# In DOLFINx, we instead rely on supplying the users with the appropriate data types for the backend. Then the user can access all of the features of the backend, not being constrained to the layer added in DOLFIN. One can also leverage the detailed documentation of PETSc. For block systems, see: https://petsc.org/release/docs/manual/ksp/?highlight=matnest#solving-block-matrices
# -

def create_block_solver(A, b, P, comm):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A, P)
    ksp.setType("minres")
    ksp.setTolerances(rtol=1e-9)
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

    nested_IS = P.getNestISs()
    ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]),
                                ("p", nested_IS[0][1]))

    # Set the preconditioners for each block
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("gamg")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("jacobi")

    # Monitor the convergence of the KSP
    ksp.setFromOptions()
    return ksp


# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Compute error estimates

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# In DOLFINx, assembling a scalar value does require any MPI-communication. `dolfinx.fem.petsc.assemble_scalar` will only integrate over the cells owned by the process.
# It is up to the user to gather the results, which can be gathered on for instance all processes, or a single process for outputting.

# +
def assemble_scalar(J, comm: MPI.Comm):
    scalar_form = fem.form(J)
    local_J = fem.assemble_scalar(scalar_form)
    return comm.allreduce(local_J, op=MPI.SUM)

def compute_errors(u, p):
    domain = u.function_space.mesh
    x = SpatialCoordinate(domain)
    error_u = u - u_ex(x)
    H1_u = inner(error_u, error_u) * dx \
         + inner(grad(error_u), grad(error_u)) * dx
    velocity_error = np.sqrt(assemble_scalar(H1_u, domain.comm))

    error_p = -p - p_ex(x)
    L2_p = fem.form(error_p * error_p * dx)
    pressure_error = np.sqrt(assemble_scalar(L2_p, domain.comm))
    return velocity_error, pressure_error


# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Solving the Stokes problem with a block-preconditioner
# -

def solve_stokes(u_element, p_element, domain):
    V = fem.FunctionSpace(domain, u_element)
    Q = fem.FunctionSpace(domain, p_element)
    
    lhs_form = create_bilinear_form(V, Q)
    rhs_form = create_linear_form(V, Q)
    
    bcs = create_velocity_bc(V)
    nsp = create_nullspace(rhs_form)
    A, b = assemble_system(lhs_form, rhs_form, bcs)
    assert nsp.test(A)
    A.setNullSpace(nsp)

    P = create_preconditioner(Q, lhs_form, bcs)
    ksp = create_block_solver(A, b, P, domain.comm)
 
    u, p = fem.Function(V), fem.Function(Q)
    w = PETSc.Vec().createNest([u.vector, p.vector])
    ksp.solve(b, w)
    assert ksp.getConvergedReason() > 0
    u.x.scatter_forward()
    p.x.scatter_forward()
    return compute_errors(u, p)


# + [markdown] tags=[] slideshow={"slide_type": "notes"}
# We now use the Stokes solver we have defined to experiment with a range of element pairs that can be used. First, we define a function that takes an element as input and plots a graph showing the error as $h$ is decreased.
# + tags=[] slideshow={"slide_type": "skip"}
def error_plot(element_u, element_p, convergence_u=None,
               convergence_p=None, refinements=5, N0=7):
    hs = np.zeros(refinements)
    u_errors = np.zeros(refinements)
    p_errors = np.zeros(refinements)
    comm = MPI.COMM_WORLD
    for i in range(refinements):
        N = N0 * 2**i
        domain = mesh.create_unit_square(comm, N, N, 
                                         cell_type=mesh.CellType.triangle)
        u_errors[i] , p_errors[i] = solve_stokes(element_u, element_p, domain)
        hs[i] = 1. / N
    legend = []
    
    if convergence_u is not None:
        y_value = u_errors[-1] * 1.4
        plt.plot([hs[0], hs[-1]], [y_value * (hs[0] / hs[-1])**convergence_u, y_value], "k--")
        legend.append(f"order {convergence_u}")
    if convergence_p is not None:
        y_value = p_errors[-1] * 1.4
        plt.plot([hs[0], hs[-1]], [y_value * (hs[0] / hs[-1])**convergence_p, y_value], "m--")
        legend.append(f"order {convergence_p}")

    plt.plot(hs, u_errors, "bo-")
    plt.plot(hs, p_errors, "ro-")
    legend += [r"$H^1(\mathbf{u_h}-\mathbf{u}_{ex})$", r"$L^2(p_h-p_ex)$"]
    plt.legend(legend)
    plt.xscale("log")
    plt.yscale("log")
    plt.axis("equal")
    plt.ylabel("Error in energy norm")
    plt.xlabel("$h$")
    plt.xlim(plt.xlim()[::-1])
    plt.grid(True)
# + [markdown] tags=[] slideshow={"slide_type": "slide"}
# ## Piecewise constant pressure spaces

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# For our first element, we pair piecewise linear elements with piecewise constants.
# -

# <img src='./img/element-Lagrange-triangle-1-dofs-large.png' style='width:100px' /><img src='./img/element-Lagrange-triangle-0-dofs-large.png' style='width:100px' />

# + [markdown] slideshow={"slide_type": "skip"} tags=[]
# Using these elements, we do not converge to the solution.
# -

element_u = VectorElement("Lagrange", "triangle", 1)
element_p = FiniteElement("DG", "triangle", 0)
error_plot(element_u, element_p)

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## P2-DG0 (Fortin, 1972)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# One way to obtain convergence with a piecewise constant pressure space is to use a piecewise quadratic space for the velocity (Fortin, 1972).
# -

# <img src='./img/element-Lagrange-triangle-2-dofs-large.png' style='width:100px' /><img src='./img/element-Lagrange-triangle-0-dofs-large.png' style='width:100px' />

element_u = VectorElement("Lagrange", "triangle", 2)
element_p = FiniteElement("DG", "triangle", 0)
error_plot(element_u, element_p, 2, 1)


# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Crouzeix-Raviart

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# Alternatively, the same order convergence can be achieved using fewer degrees of freedom if a Crouzeix-Raviart element is used for the velocity space (Crouziex, Raviart, 1973).
# -

# <img src='./img/element-Crouzeix-Raviart-triangle-1-dofs-large.png' style='width:100px' /><img src='./img/element-Lagrange-triangle-0-dofs-large.png' style='width:100px' />

element_u = VectorElement("CR", "triangle", 1)
element_p = FiniteElement("DG", "triangle", 0)
error_plot(element_u, element_p, 1)

# + [markdown] tags=[] slideshow={"slide_type": "slide"}
# ## Piecewise linear pressure space (Crouziex, Falk, 1988)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# When using a piecewise linear pressure space, we could again try using a velocity space one degree higher, but we would again observe that there is no convergence. In order to achieve convergence, we can augment the quadratic space with a cubic bubble function on the triangle (Crouziex, Falk, 1988).
# -

# <img src='./img/element-bubble-enriched-Lagrange-triangle-2-dofs-large.png' style='width:100px' /><img src='./img/element-Lagrange-triangle-1-dofs-large.png' style='width:100px' />

enriched_element = FiniteElement("Lagrange", "triangle", 2) \
                 + FiniteElement("Bubble", "triangle", 3)
element_u = VectorElement(enriched_element)
element_p = FiniteElement("DG", "triangle", 1)
error_plot(element_u, element_p, 2)

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## Piecewise quadratic pressure space (Crouzeix, Falk, 1988)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# When using a piecewise quadratic space, we can use a cubic velocity space augmented with quartic bubbles (Crouzeix, Falk, 1988).
# -

# <img src='./img/element-bubble-enriched-Lagrange-triangle-3-dofs-large.png' style='width:100px' /><img src='./img/element-Lagrange-triangle-2-dofs-large.png' style='width:100px' />

# ### Custom elements

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# We have to define this velocity element as a custom element (it cannot be created as an enriched element, as the basis functions of degree 3 Lagrange and degree 4 bubbles are not linearly independent). More examples of how custom elements can be created can be found [in the Basix documentation](https://docs.fenicsproject.org/basix/v0.5.0/python/demo/demo_custom_element.py.html).

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# #### Defining the polynomial space
#
# When creating a custom element, we must input the coefficients that define a basis of the set of polynomials that our element spans. In this example, we will represent the 9 functions in our space in terms of the 10 orthonormal polynomials of degree $\leqslant3$ on a quadrilateral, so we create a 9 by 10 matrix.
#
# A polynomial $f$ on the triangle can be written as
# $$f(x,y)=\sum_i\left(\int_0^1\int_0^{1-y}f(x, y)q_i(x, y) \,\mathrm{d}x\,\mathrm{d}y\right)q_i(x,y),$$
# where $q_0$, $q_1$, ... are the orthonormal polynomials. The entries of our coefficient matrix are these integrals.

# + slideshow={"slide_type": "notes"} tags=[]
wcoeffs = np.zeros((12, 15))
pts, wts = basix.make_quadrature(basix.CellType.triangle, 8)
poly = basix.tabulate_polynomials(basix.PolynomialType.legendre, basix.CellType.triangle, 4, pts)
x = pts[:, 0]
y = pts[:, 1]
for j, f in enumerate([
    1, x, y, x**2*y, x*y**2, (1-x-y)**2*y, (1-x-y)*y**2, x**2*(1-x-y), x*(1-x-y)**2,
    x*y*(1-x-y), x**2*y*(1-x-y), x*y**2*(1-x-y)
]):
    for i in range(15):
        wcoeffs[j, i] = sum(f * poly[i, :] * wts)

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# #### Interpolation
#
# Next, we compute the points and matrices that define how functions can be interpolated into this space. For this element, all the DOFs are point evaluations, so we create lists of these points and (reshaped) identity matrices.

# + slideshow={"slide_type": "notes"} tags=[]
x = [[], [], [], []]
x[0].append(np.array([[0.0, 0.0]]))
x[0].append(np.array([[1.0, 0.0]]))
x[0].append(np.array([[0.0, 1.0]]))
x[1].append(np.array([[2 / 3, 1 / 3], [1 / 3, 2 / 3]]))
x[1].append(np.array([[0.0, 1 / 3], [0.0, 2 / 3]]))
x[1].append(np.array([[1 / 3, 0.0], [2 / 3, 0.0]]))
x[2].append(np.array([[1 / 4, 1 / 4], [1 / 2, 1 / 4], [1 / 4, 1 / 2]]))

M = [[], [], [], []]
for _ in range(3):
    M[0].append(np.array([[[[1.]]]]))
for _ in range(3):
    M[1].append(np.array([[[[1.], [0.]]], [[[0.], [1.]]]]))
M[2].append(np.array([[[[1.], [0.], [0.]]], [[[0.], [1.], [0.]]], [[[0.], [0.], [1.]]]]))

# + [markdown] slideshow={"slide_type": "notes"} tags=[]
# #### Creating the element
#
# We now create the element by passing in the information we created above, as well as the cell type, value shape, number of derivatives used by the DOFs, map type, whether the element is discontinuous, the highest degree Lagrange space that is a subspace of the element, and the polynomial degree of the element.
# -

p3_plus_bubbles = basix.create_custom_element(basix.CellType.triangle, [], wcoeffs, x, M, 0, basix.MapType.identity, False, 3, 4)
element_u = VectorElement(basix.ufl_wrapper.BasixElement(p3_plus_bubbles))
element_p = FiniteElement("DG", "triangle", 2)
error_plot(element_u, element_p, 3, refinements=4)

# + [markdown] slideshow={"slide_type": "slide"} tags=[]
# ## References
#
# Crouzeix, Michel and Falk, Richard S. Nonconforming finite elements for the Stokes problem, *Mathematics of Computation* 52, 437–456, 1989. [DOI: [10.2307/2008475](https://doi.org/10.2307/2008475)]
#
# Crouzeix, Michel and Raviart, Pierre-Arnaud. Conforming and nonconforming finite element methods for solving the stationary Stokes equations, *Revue Française d'Automatique, Informatique et Recherche Opérationnelle* 3, 33–75, 1973. [DOI: [10.1051/m2an/197307R300331](https://doi.org/10.1051/m2an/197307R300331)]
#
# Fortin, Michel. Calcul numérique des écoulements des fluides de Bingham et des fluides newtoniens incompressibles par la méthode des éléments finis (PhD thesis), Univ. Paris, 1972.
#
# The images of elements used in this example were taken from DefElement:<br />
# The DefElement contributors. DefElement: an encyclopedia of finite element definitions, 2022, https://defelement.com [Online; accessed: 15-August-2022].
# -


