# fenics22-tutorial

Tutorial for the FEniCS [22 conference](https://fenicsproject.org/fenics-2022/) in San Diego


## Adding a tutorial to the book

Add a chapter to `_toc.yml`.

Inside the Jupyter notebook, go to `Property Inspector` (the two cogwheels in the top right corner of JupyterLab)
and add the following as notebook metadata:
```yml
 {
  "jupytext": {
   "formats": "ipynb,py:light"
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.4"
  }
 }
```
This will choose the default kernel in the `dolfinx/lab` docker image, and automatically convert the notebooks to a `.py` file at saving.

If you want to use complex numbers, change:
```bash
 "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
```
to
```bash
  "kernelspec": {
   "display_name": "Python 3 (DOLFINx complex)",
   "language": "python",
   "name": "python3-complex"
  },
```


## Create slides from your notebook

You can use `nbconvert` (`pip3 install nbconvert`) to convert the `.ipynb` to a presentation.
The command to run is:
```bash
jupyter nbconvert example.ipynb --to html --template reveal
```

To change what is rendered on each slide, you can change the notebook metadata,
which is in `Property Inspector` (the two cogwheels in the top right corner of JupyterLab), and change the `Slide Type` to `Slide` to start a new slide. If you want to add the cell below to the same slide, change the type to `-`.

If a cell should be revealed with `Right Arrow`, choose `Fragment`.

If you want a sub-slide, i.e. navigating downwards with arrows when rendering the presentation, change the type to `Sub-Slide`.

If a cell should be ignored in presentation mode, set it to `Notes`.

## Hiding cells/output
See https://jupyterbook.org/en/stable/interactive/hiding.htm for more details. The setting is also in advanced tools on the RHS of the Jupyterlab interface
### Automatically generate slides
By adding the following file to the (`jupyter_server_config.py`) `.jupyter` folder on your system. 
You might need to rename it to `jupyter_notebook_config.py`.
To check the config paths, call:
```bash
jupyter server --show-config
jupyter notebook --show-config
```

If you run the code with `dolfinx/lab:nightly` using for instance:
```bash
docker run -ti -p 8888:8888 --rm -v $(pwd):/root/shared -w /root/shared dokken92/dolfinx_custom:15072022
```
no copying is required.


# Render the HTML files directly of Github
Use [githack](https://raw.githack.com/) and add the link to the relevant presentation.

Example:
- [Example page](https://raw.githack.com/jorgensd/fenics22-tutorial/main/part1/example.html#/)
- [Time dependent problem](https://raw.githack.com/jorgensd/fenics22-tutorial/main/part1/heat_eq.html#/)
- [Helmholtz](https://raw.githack.com/jorgensd/fenics22-tutorial/main/part2/helmholtz.html#/)
- [Stokes](https://raw.githack.com/jorgensd/fenics22-tutorial/main/part3-stokes/comparing_elements.html#/)