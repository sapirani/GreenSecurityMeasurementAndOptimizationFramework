import os

from matplotlib import pyplot as plt


def is_results_dir(dir_name: str) -> bool:
    return dir_name is not None and dir_name.startswith("results")

