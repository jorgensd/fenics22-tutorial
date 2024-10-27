import os
from subprocess import check_call

from traitlets.config import Config


def post_save(model, os_path, contents_manager):
    """post-save hook for converting notebooks to .py scripts"""
    print(model["type"])
    if model["type"] != "notebook":
        return  # only do this for notebooks
    d, fname = os.path.split(os_path)
    if ".py" in fname:
        return  # Ignore python files

    check_call(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            "--TagRemovePreprocessor.remove_input_tags={'remove-input', 'hide-input'}",
            fname,
            "--template",
            "reveal",
            f"--output=presentation-{fname.split('.ipynb')[0]}",
        ],
        cwd=d,
    )


c = Config()
c.FileContentsManager.post_save_hook = post_save
