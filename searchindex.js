Search.setIndex({"docnames": ["access", "comparing_elements", "example", "heat_eq", "helmholtz", "intro"], "filenames": ["access.md", "comparing_elements.ipynb", "example.ipynb", "heat_eq.ipynb", "helmholtz.ipynb", "intro.md"], "titles": ["Further information", "The Stokes equations", "Introduction to DOLFINx", "Solving a time-dependent problem", "The Helmholtz equation", "FEniCSx Tutorial &#64; FEniCS 2022"], "terms": {"The": [0, 3, 5], "can": [0, 1, 2, 3, 4], "found": [0, 1], "jorgensd": [0, 5], "github": [0, 2], "io": [0, 3, 4], "fenics22": 0, "document": [0, 1], "all": [0, 1, 3, 4], "packag": [0, 1], "doc": [0, 1], "fenicsproject": 0, "org": [0, 1], "A": [0, 1, 3, 5], "larg": 0, "collect": 0, "code": [0, 1], "exampl": [0, 1, 3, 4], "question": 0, "about": 0, "us": [0, 1, 3], "post": 0, "fenic": [0, 2], "discours": 0, "forum": 0, "If": [0, 3], "you": [0, 1, 2, 3], "wish": 0, "j\u00f8rgen": [0, 5], "s": [0, 1, 5], "dokken": [0, 1, 5], "igor": [0, 5], "baratta": [0, 5], "joseph": [0, 5], "dean": [0, 5], "sarah": [0, 5], "roggendorf": [0, 1, 5], "matthew": [0, 5], "w": [0, 1, 3, 4, 5], "scrogg": [0, 1, 5], "david": [0, 5], "kamenski": [0, 5], "adeeb": [0, 5], "arif": [0, 5], "kor": [0, 5], "michal": [0, 5], "habera": [0, 5], "chri": [0, 5], "richardson": [0, 5], "nathan": [0, 5], "sime": [0, 5], "2022": [0, 1], "avail": 0, "onlin": [0, 1], "bibtex": 0, "citat": 0, "misc": 0, "fenics2022tutori": 0, "author": [0, 1], "j": [0, 1, 4], "o": 0, "rgen": 0, "p": [0, 1], "titl": 0, "year": 0, "howpublish": 0, "url": 0, "http": [0, 1, 2], "note": [0, 3, 4], "access": [0, 1], "22": [0, 2, 3, 4], "august": [0, 1], "m": 1, "begin": [1, 3, 4], "align": [1, 3, 4], "delta": [1, 3, 4], "mathbf": [1, 4], "u": [1, 3, 4], "nabla": [1, 3, 4], "f": [1, 2, 3, 4], "text": [1, 3, 4], "omega": [1, 3, 4], "cdot": [1, 3, 4], "0": [1, 2, 3, 4], "partial": [1, 3, 4], "end": [1, 3, 4], "In": [1, 2, 3, 4], "thi": [1, 3, 4], "tutori": [1, 4], "learn": [1, 4], "how": [1, 3, 4], "ufl": [1, 3, 4], "we": [1, 2, 3, 4], "start": [1, 2, 3], "most": [1, 3], "modul": [1, 2, 3], "from": [1, 2, 3, 4], "dolfinx": [1, 3, 4, 5], "fem": [1, 3, 4], "mesh": [1, 4], "mpi4pi": [1, 2, 3, 4], "mpi": [1, 2, 3, 4], "petsc4pi": [1, 3, 4], "basix": 1, "ufl_wrapp": 1, "matplotlib": [1, 3, 4], "pylab": 1, "plt": [1, 3, 4], "numpi": [1, 3, 4], "np": [1, 3, 4], "warn": 1, "filterwarn": 1, "ignor": 1, "howev": [1, 3], "pai": 1, "special": 1, "attent": 1, "unifi": 1, "languag": 1, "which": [1, 3], "repres": 1, "As": [1, 3, 4], "dependend": 1, "mani": [1, 3], "function": [1, 3, 4], "compon": 1, "explicitli": 1, "vectorel": 1, "finiteel": [1, 4], "spatialcoordin": [1, 3, 4], "trialfunct": [1, 3, 4], "testfunct": [1, 3, 4], "as_vector": [1, 3], "co": [1, 3, 5], "sin": 1, "inner": [1, 3, 4], "div": 1, "grad": [1, 3, 4], "dx": [1, 3, 4], "pi": [1, 3, 4], "known": 1, "analyt": 1, "exact": 1, "veloc": 1, "follow": 1, "def": [1, 3, 4], "u_ex": 1, "x": [1, 3, 4], "sinx": 1, "sini": 1, "cosx": 1, "cosi": 1, "c_factor": 1, "2": [1, 3, 4], "return": [1, 3], "p_ex": 1, "here": 1, "input": [1, 3, 4], "each": 1, "coordin": [1, 3], "y": [1, 3], "These": 1, "domain": [1, 4], "strong": 1, "formul": 1, "pde": [1, 4], "sourc": [1, 3], "oper": [1, 3], "set": [1, 2], "one": [1, 4], "lead": 1, "discret": [1, 4], "b": [1, 3], "pmatrix": 1, "a_": 1, "create_bilinear_form": 1, "v": [1, 3, 4], "q": 1, "a_uu": 1, "a_up": 1, "a_pu": 1, "none": 1, "create_linear_form": 1, "create_velocity_bc": 1, "g": [1, 4], "tdim": [1, 3], "topolog": [1, 2, 3, 4], "dim": [1, 3], "create_connect": 1, "bdry_facet": 1, "exterior_facet_indic": 1, "dof": [1, 3], "locate_dofs_topolog": [1, 3], "dirichletbc": [1, 3], "descript": 1, "abov": [1, 3], "have": [1, 2, 3, 4], "onli": [1, 3], "ad": 1, "mean": 1, "singular": 1, "ie": [1, 2, 3], "determin": 1, "up": 1, "therefor": [1, 3], "nullspac": 1, "create_nullspac": 1, "rhs_form": 1, "null_vec": 1, "create_vector_nest": 1, "getnestsubvec": 1, "normal": 1, "nsp": 1, "vector": [1, 3], "matrix": [1, 3], "top": 1, "left": [1, 3], "bottom": 1, "right": [1, 3], "diagon": 1, "entri": 1, "mass": 1, "create_precondition": 1, "bc": [1, 3], "a_p11": 1, "a_p": 1, "assemble_matrix_nest": 1, "assemble_system": 1, "lhs_form": 1, "assemble_vector_nest": 1, "apply_lifting_nest": 1, "b_sub": 1, "ghostupd": 1, "addv": 1, "insertmod": 1, "add": [1, 2, 3], "mode": [1, 2], "scattermod": [1, 3], "revers": 1, "extract_function_spac": 1, "bcs0": 1, "bcs_by_block": 1, "set_bc_nest": 1, "legaci": [1, 2, 3], "dolfin": [1, 2, 3], "conveni": [1, 3], "were": 1, "provid": 1, "interact": [1, 3], "algebra": [1, 3], "instead": [1, 2], "suppli": [1, 3], "user": [1, 3], "appropri": [1, 3, 4], "data": [1, 3, 4], "type": [1, 3], "so": [1, 3], "featur": 1, "rather": 1, "than": 1, "being": 1, "constrain": 1, "our": [1, 3], "wrapper": 1, "One": 1, "also": [1, 3], "leverag": 1, "detail": 1, "For": [1, 3], "see": [1, 4], "releas": 1, "manual": 1, "ksp": [1, 3], "highlight": [1, 3], "matnest": 1, "matric": [1, 3], "create_block_solv": 1, "comm": [1, 3, 4], "setoper": [1, 3], "settyp": [1, 3], "minr": 1, "settoler": 1, "rtol": 1, "1e": 1, "9": 1, "getpc": [1, 3], "fieldsplit": 1, "setfieldsplittyp": 1, "pc": [1, 3], "compositetyp": 1, "addit": 1, "nested_i": 1, "getnestiss": 1, "setfieldspl": 1, "ksp_u": 1, "ksp_p": 1, "getfieldsplitsubksp": 1, "preonli": [1, 4], "gamg": 1, "jacobi": 1, "monitor": 1, "converg": 1, "setfromopt": 1, "scalar": [1, 3], "valu": [1, 3, 4], "doe": 1, "requir": [1, 4], "ani": [1, 3], "commun": [1, 3, 4], "assemble_scalar": 1, "integr": [1, 3], "over": [1, 3, 4], "cell": [1, 2, 3, 4], "own": 1, "process": 1, "It": 1, "gather": 1, "result": 1, "instanc": 1, "singl": [1, 3], "output": [1, 3], "scalar_form": 1, "local_j": 1, "allreduc": 1, "op": 1, "sum": 1, "compute_error": 1, "function_spac": 1, "error_u": 1, "h1_u": 1, "velocity_error": 1, "sqrt": [1, 4], "error_p": 1, "l2_p": 1, "pressure_error": 1, "solve_stok": 1, "u_el": 1, "p_element": 1, "functionspac": [1, 3, 4], "assert": 1, "test": [1, 3], "setnullspac": 1, "vec": 1, "createnest": 1, "getconvergedreason": 1, "scatter_forward": [1, 3], "now": [1, 3], "experi": 1, "rang": 1, "pair": 1, "first": [1, 4], "take": [1, 3], "plot": [1, 4], "graph": 1, "show": [1, 3], "h": 1, "decreas": 1, "error_plot": 1, "element_u": 1, "element_p": 1, "convergence_u": 1, "convergence_p": 1, "refin": 1, "5": [1, 2, 3], "n0": 1, "7": 1, "hs": 1, "zero": 1, "u_error": 1, "p_error": 1, "comm_world": [1, 2, 3, 4], "i": 1, "n": [1, 3, 4], "create_unit_squar": [1, 2], "celltyp": [1, 3], "triangl": 1, "legend": 1, "y_valu": 1, "4": [1, 4], "k": [1, 4], "append": 1, "order": [1, 4], "bo": 1, "ro": 1, "r": 1, "u_h": 1, "_": [1, 4], "ex": 1, "l": [1, 3, 4], "p_h": 1, "xscale": 1, "log": 1, "yscale": 1, "axi": 1, "equal": 1, "ylabel": 1, "energi": 1, "norm": 1, "xlabel": 1, "xlim": 1, "grid": [1, 2, 3, 4], "true": [1, 2, 3, 4], "do": [1, 2, 3], "lagrang": [1, 3, 4], "dg": [1, 4], "wai": [1, 3], "obtain": [1, 3], "altern": 1, "same": 1, "achiev": 1, "fewer": 1, "degre": [1, 3, 4], "freedom": [1, 3, 4], "cr": 1, "when": [1, 3, 4], "could": 1, "again": 1, "try": 1, "higher": [1, 4], "would": [1, 3], "observ": 1, "augment": 1, "cubic": 1, "bubbl": 1, "enriched_el": 1, "continu": 1, "cg": [1, 3], "quartic": 1, "cannot": 1, "an": [1, 3, 4], "enrich": 1, "basi": 1, "ar": [1, 2, 3, 4], "linearli": 1, "independ": [1, 3], "more": [1, 3, 4], "must": 1, "coeffici": 1, "span": [1, 3], "term": 1, "10": [1, 2, 3, 4], "orthonorm": 1, "leqslant3": 1, "quadrilater": [1, 3], "written": [1, 3], "sum_i": 1, "int_0": 1, "q_i": 1, "mathrm": [1, 4], "d": [1, 3], "where": [1, 3, 4], "q_0": 1, "q_1": 1, "wcoeff": 1, "12": [1, 3], "15": [1, 4], "pt": 1, "wt": 1, "make_quadratur": 1, "8": [1, 3, 4], "poli": 1, "tabulate_polynomi": 1, "polynomialtyp": 1, "legendr": 1, "enumer": 1, "next": [1, 3, 4], "point": [1, 2, 3, 4], "evalu": [1, 3, 4], "list": [1, 3], "reshap": 1, "ident": 1, "arrai": [1, 3, 4], "pass": [1, 4], "inform": [1, 4, 5], "well": [1, 3], "shape": 1, "number": [1, 4], "deriv": 1, "map": [1, 4], "whether": 1, "discontinu": [1, 3, 4], "highest": 1, "p3_plus_bubbl": 1, "create_custom_el": 1, "maptyp": 1, "fals": [1, 3, 4], "basixel": 1, "imag": 1, "taken": 1, "defel": 1, "michel": 1, "fortin": 1, "calcul": 1, "num\u00e9riqu": 1, "de": 1, "\u00e9coulement": 1, "fluid": 1, "bingham": 1, "et": 1, "newtonien": 1, "incompress": 1, "par": 1, "la": [1, 3], "m\u00e9thode": 1, "\u00e9l\u00e9ment": 1, "fini": 1, "phd": 1, "thesi": 1, "univ": 1, "pari": 1, "1972": 1, "ichel": 1, "c": 1, "rouzeix": 1, "ierr": 1, "rnaud": 1, "aviart": 1, "onform": 1, "nonconform": 1, "finit": [1, 3], "method": [1, 3], "stationari": 1, "toke": 1, "revu": 1, "fran": 1, "\u00e7": 1, "ais": 1, "automatiqu": 1, "informatiqu": 1, "recherch": 1, "op\u00e9rationnel": 1, "33": 1, "75": 1, "1973": 1, "doi": 1, "1051": 1, "m2an": 1, "197307r300331": 1, "ichard": 1, "alk": 1, "onconform": 1, "mathemat": 1, "52": 1, "437": 1, "456": 1, "1989": 1, "2307": 1, "2008475": 1, "contributor": 1, "ef": 1, "e": [1, 4], "lement": 1, "encyclopedia": 1, "definit": 1, "com": [1, 2, 5], "import": [2, 3, 4], "check": 2, "version": [2, 3], "git": 2, "commit": 2, "hash": 2, "print": [2, 4], "__version__": 2, "instal": [2, 4], "base": [2, 3], "nhttp": 2, "common": 2, "git_commit_hash": 2, "2aaf3b20dbaedcbd3925a9640c3859deec563e02": 2, "wildcard": 2, "pyvista": [2, 3], "geometri": [2, 3, 4], "create_vtk_mesh": [2, 3, 4], "unstructuredgrid": [2, 3, 4], "both": [2, 3], "static": 2, "start_xvfb": [2, 3, 4], "set_jupyter_backend": 2, "pythreej": [2, 3, 4], "plotter": [2, 3, 4], "window_s": [2, 3, 4], "600": 2, "render": [2, 3, 4], "add_mesh": [2, 3, 4], "show_edg": [2, 3, 4], "present": 2, "view_xi": [2, 3, 4], "camera": [2, 3, 4], "zoom": [2, 3, 4], "1": [2, 3, 4], "35": 2, "export_html": [2, 3, 4], "html": [2, 3, 4], "backend": [2, 3, 4], "0m": [2, 3, 4], "2m2024": [2, 3, 4], "01": [2, 3, 4], "08": [2, 3, 4], "05": [2, 3, 4], "18": 2, "306": 2, "888": 2, "a76e2000": 2, "vtkextractedg": [2, 3, 4], "cxx": [2, 3, 4], "435": [2, 3, 4], "info": [2, 3, 4], "0mexecut": [2, 3, 4], "edg": [2, 3, 4], "extractor": [2, 3, 4], "renumb": [2, 3, 4], "307": 2, "889": 2, "551": [2, 3, 4], "0mcreat": [2, 3, 4], "320": 2, "323": 2, "905": 2, "get": 2, "notebook": [2, 3], "call": 2, "ifram": [2, 3, 4], "src": [2, 3, 4], "width": [2, 3, 4], "610px": 2, "height": [2, 3, 4], "noqa": [2, 3, 4], "transient": 3, "differ": [3, 4], "between": 3, "look": 3, "structur": 3, "relev": 3, "class": [3, 4], "relat": 3, "element": [3, 4], "read": [3, 4], "write": [3, 4], "export": 3, "linear": 3, "To": [3, 4], "simpl": 3, "gener": [3, 4], "util": 3, "tool": 3, "build": [3, 4], "rectangl": 3, "triangular": 3, "box": 3, "tetrahedr": 3, "hexahedr": 3, "3": [3, 4], "100": 3, "20": [3, 4], "direct": [3, 4], "respect": 3, "length": 3, "nx": 3, "ny": 3, "80": 3, "60": 3, "extent": 3, "create_rectangl": 3, "constrast": 3, "work": 3, "python": [3, 4], "nest": 3, "etc": 3, "send": 3, "becaus": 3, "want": 3, "awar": 3, "run": 3, "parallel": 3, "comm_self": 3, "initialis": 3, "script": 3, "full": 3, "local": [3, 4], "its": [3, 4], "local_domain": 3, "With": 3, "format": 3, "includ": [3, 4], "png": 3, "beam": 3, "924": 3, "377": 3, "b119d000": 3, "927": 3, "381": 3, "9740": 3, "949": 3, "402": 3, "952": 3, "405": 3, "scroll": 3, "800px": 3, "400px": 3, "heat": 3, "equat": [3, 5], "backward": 3, "euler": 3, "step": 3, "scheme": 3, "frac": 3, "u_": [3, 4], "u_n": 3, "t": 3, "mu": 3, "t_": 3, "u_d": 3, "omega_": 3, "25t": 3, "defin": 3, "hand": 3, "remain": 3, "three": 3, "space": [3, 4], "correspond": [3, 4], "trial": 3, "materi": 3, "tempor": 3, "paramet": 3, "explicit": 3, "avoid": 3, "confus": 3, "thei": 3, "origin": 3, "support": [3, 4], "real": [3, 4], "complex": [3, 4], "abl": [3, 4], "petsc": [3, 4], "float": 3, "compil": 3, "need": 3, "form": 3, "ensur": 3, "consist": 3, "system": 3, "un": 3, "constant": [3, 4], "dt": 3, "syntax": 3, "done": [3, 4], "There": 3, "dimension": 3, "column": 3, "z": 3, "directli": 3, "ud_funct": 3, "lambda": 3, "25": [3, 4], "ud": 3, "interpol": 3, "give": 3, "few": 3, "locate_dofs_geometr": 3, "locat": 3, "advis": 3, "certain": 3, "geometr": 3, "associ": 3, "them": 3, "eg": 3, "n\u00e9d\u00e9lec": 3, "raviart": 3, "thoma": 3, "ha": [3, 4], "convenic": 3, "facet": 3, "dirichlet_facet": 3, "isclos": 3, "bc_facet": 3, "locate_entities_boundari": 3, "bndry_dof": 3, "side": 3, "like": 3, "re": 3, "assembl": 3, "everi": 3, "control": 3, "onc": 3, "outsid": 3, "loop": 3, "compiled_a": 3, "assemble_matrix": 3, "kernel": 3, "rh": 3, "compiled_l": 3, "krylov": 3, "subspac": 3, "multigrid": 3, "hypr": 3, "sethypretyp": 3, "boomeramg": 3, "anim": 3, "solut": 3, "vtk": 3, "compat": 3, "pyplot": [3, 4], "virtual": 3, "framebuff": 3, "open_gif": 3, "u_tim": 3, "gif": 3, "uh": [3, 4], "point_data": [3, 4], "viridi": 3, "cm": 3, "get_cmap": 3, "sarg": [3, 4], "dict": [3, 4], "title_font_s": [3, 4], "label_font_s": [3, 4], "fmt": [3, 4], "2e": [3, 4], "color": [3, 4], "black": [3, 4], "position_x": [3, 4], "position_i": [3, 4], "light": 3, "cmap": 3, "scalar_bar_arg": [3, 4], "clim": 3, "readi": 3, "At": 3, "updat": 3, "reassembl": 3, "appli": 3, "au": 3, "previou": 3, "while": 3, "assemble_vector": 3, "apply_lift": 3, "scatter_revers": 3, "set_bc": 3, "update_scalar": 3, "write_fram": 3, "close": 3, "express": [3, 4], "scalabl": 3, "case": 3, "introduc": 3, "given": 3, "refer": 3, "let": 3, "consid": 3, "x_grad": 3, "dq": 3, "expr": 3, "interpolation_point": 3, "w_grid": 3, "w_plotter": 3, "800": 3, "200": 3, "31": 3, "649": 3, "103": 3, "653": 3, "106": 3, "19200": 3, "678": 3, "131": 3, "682": 3, "135": 3, "200px": 3, "solv": [4, 5], "field": 4, "high": 4, "subject": 4, "absorb": 4, "condit": 4, "ku": 4, "piecewis": 4, "wavenumb": 4, "comput": 4, "inc": 4, "ku_": 4, "incom": 4, "plane": 4, "wave": 4, "design": 4, "execut": 4, "sy": 4, "issubdtyp": 4, "scalartyp": 4, "complexflo": 4, "exit": 4, "els": 4, "complex128": 4, "free": 4, "air": 4, "k0": 4, "wavelength": 4, "lmbda": 4, "polynomi": 4, "6": 4, "mesh_ord": 4, "long": 4, "been": 4, "api": 4, "turn": 4, "distribut": 4, "gmshio": 4, "model_to_mesh": 4, "generate_mesh": 4, "creat": 4, "rank": 4, "mesh_gener": 4, "file_nam": 4, "msh": 4, "cell_tag": 4, "read_from_msh": 4, "gdim": 4, "entiti": 4, "2985": 4, "node": 4, "1444": 4, "part": 4, "depend": [4, 5], "marker": 4, "through": 4, "fact": 4, "wise": 4, "find": 4, "set_plot_them": 4, "export_funct": 4, "name": 4, "show_mesh": 4, "tessel": 4, "set_active_scalar": 4, "700": 4, "t_grid": 4, "grid_mesh": 4, "style": 4, "wirefram": 4, "line_width": 4, "cell_data": 4, "37": 4, "272": 4, "037": 4, "c1a81000": 4, "273": 4, "039": 4, "2214": 4, "292": 4, "058": 4, "293": 4, "059": 4, "700px": 4, "jkx": 4, "propag": 4, "quantiti": 4, "quadratur": 4, "facetnorm": 4, "uinc": 4, "exp": 4, "1j": 4, "dot": 4, "4th": 4, "product": 4, "int_": 4, "bar": 4, "ds": 4, "qquad": 4, "foral": 4, "widehat": 4, "ufl_cel": 4, "lu": 4, "opt": 4, "ksp_type": 4, "pc_type": 4, "linearproblem": 4, "petsc_opt": 4, "ab": 4, "u_ab": 4, "dtype": 4, "float64": 4, "xdmf": 4, "out": 4, "file": 4, "write_mesh": 4, "write_funct": 4, "vtx": 4, "out_high_ord": 4, "bp": 4, "univers": 5, "cambridg": 5, "igorbaratta": 5, "jpdean": 5, "jsdokken": 5, "sarahro": 5, "mscrogg": 5, "uk": 5, "california": 5, "san": 5, "diego": 5, "universit\u00e9": 5, "du": 5, "luxembourg": 5, "carnegi": 5, "institut": 5, "scienc": 5, "introduct": 5, "time": 5, "problem": 5, "helmholtz": 5, "stoke": 5, "further": 5}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"further": 0, "inform": 0, "fenicsx": [0, 5], "help": 0, "cite": 0, "thi": 0, "tutori": [0, 5], "content": [0, 5], "The": [1, 4], "stoke": 1, "equat": [1, 4], "import": 1, "defin": [1, 4], "manufactur": 1, "solut": 1, "variat": [1, 3, 4], "form": [1, 4], "boundari": [1, 3, 4], "condit": [1, 3], "creat": [1, 3], "block": 1, "precondition": 1, "assembl": 1, "nest": 1, "system": 1, "petsc": 1, "krylov": 1, "subspac": 1, "solver": [1, 3, 4], "comput": [1, 3], "error": 1, "estim": 1, "solv": [1, 3], "problem": [1, 3, 4], "piecewis": 1, "constant": 1, "pressur": 1, "space": 1, "p2": 1, "dg0": 1, "1": 1, "crouzeix": 1, "raviart": 1, "linear": [1, 4], "taylor": 1, "hood": 1, "element": 1, "quadrat": 1, "3": 1, "custom": 1, "polynomi": 1, "interpol": 1, "refer": 1, "introduct": 2, "dolfinx": 2, "us": [2, 4], "built": 2, "mesh": [2, 3], "interfac": [2, 4], "extern": 2, "librari": 2, "interact": 2, "plot": [2, 3], "time": 3, "depend": 3, "distribut": 3, "domain": 3, "each": 3, "process": [3, 4], "set": 3, "up": 3, "dirichlet": 3, "post": [3, 4], "without": 3, "project": 3, "helmholtz": 4, "statement": 4, "model": 4, "paramet": 4, "gmsh": 4, "materi": 4, "sourc": 4, "term": 4, "postprocess": 4, "visualis": 4, "pyvista": 4, "paraview": 4, "xdmffile": 4, "vtxwriter": 4, "fenic": 5, "2022": 5, "present": 5, "contributor": 5}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})