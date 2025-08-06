import numpy as np
import matplotlib.pyplot as plt
from sktime.datasets import load_from_tsfile_to_dataframe
from pipeline import get_pipeline_from_data
from sklearn.model_selection import learning_curve
from process import process

process("input/")