import os
import argparse
import glob
import numpy as np
import pandas as pd

# Imports for plotting
# To view possible matplotlib backends use
# >>> import matplotlib
# >>> bklist = matplotlib.rcsetup.interactive_bk
# >>> print(bklist)
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('MacOSX')
# matplotlib.use('pdf')
# Get rid of type 3 fonts in figures
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# Ensure LaTeX font
font = {"family": "serif"}
plt.rc("font", **font)
plt.rcParams["figure.figsize"] = (6, 6)

df = pd.read_csv("/Users/atoivonen/Documents/repos/forks/Yoke/applications/harnesses/mnist_surrogate/runs/study_001/training_study001_epoch003.csv", names=['epoch', 'idx', 'loss'])

plt.scatter(df['idx'], df['loss'])
plt.savefig('simple_loss.png')