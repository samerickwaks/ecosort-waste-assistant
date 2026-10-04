# %% [markdown]
# # EcoSort Waste Management Assistant
# ### Module 8 Summative Lab — CNNs, transformers and retrieval-augmented generation
#
# Metro City's recovery facility pays twice for mis-sorted waste: once to move it and once to
# pick it back out of the wrong stream. EcoSort's assistant moves that decision upstream, to the
# resident holding the item, by answering one question — *which bin, and how do I prepare it?* —
# from either a photo or a sentence, and by quoting the city policy it answered from.
#
# Three models, one answer:
#
# | Part | Component | Model | Headline result on held-out data |
# |---|---|---|---|
# | 2 | Image classifier | EfficientNetB0, two-stage fine-tune (backbone chosen by a frozen-feature probe) | 0.87 accuracy, 0.87 macro F1, 0.94 top-2 over 709 photos |
# | 3 | Description classifier | TF-IDF baselines vs. fine-tuned DistilBERT | 1.00 and 0.99 — the benchmark is saturated, which §3.4 takes seriously |
# | 4 | Instruction generator | MiniLM retrieval + fine-tuned FLAN-T5 (RAG) | recall@1 1.00, grounding 0.85, 100% of citations on-stream |
# | 5 | Assistant | the three wired together with confidence gating, fusion and a feedback log | 0.83 end-to-end on photos, 0 uncaught exceptions on 11 malformed inputs |
#
# The notebook is written to be read top to bottom: each modelling choice is stated before it is
# made, every number quoted in prose is printed by the cell above it, and the four dataset
# problems found in Part 1 — class imbalance, label leakage in the text file, near-duplicate
# photographs, and ambiguous head nouns — are what the later parts are built around. Where a
# hypothesis from Part 1 fails its test later on (§3.4 is the clearest case), the notebook says
# so rather than quietly dropping it.

# %% [markdown]
# ## Part 0: Environment
#
# Two deep-learning stacks are used deliberately: **TensorFlow/Keras** for the vision model
# (Keras Applications ships ImageNet backbones with matching `preprocess_input` functions) and
# **PyTorch/Hugging Face** for the language models (DistilBERT, MiniLM, FLAN-T5). Both live in
# one kernel, so each large model is released explicitly with `release()` once it has been used;
# this machine has limited free RAM and FLAN-T5 is the largest object in the notebook.

# %%
# =========================
# Part 0: Environment
# =========================
import gc
import json
import os
import pathlib
import random
import re
import textwrap
import time
import warnings
from collections import Counter, defaultdict

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')      # keep Keras logs readable
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # TF and PyTorch both ship OpenMP on Windows
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, regularizers

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 95)
pd.set_option('display.width', 170)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


def release(*names):
    """Delete the named globals, clear the Keras graph and run a GC pass.

    Called after each large model is finished with: this kernel holds TensorFlow and PyTorch
    at the same time, and FLAN-T5 alone is ~1 GB.
    """
    for n in names:
        globals().pop(n, None)
    tf.keras.backend.clear_session()
    gc.collect()


print(f'TensorFlow {tf.__version__} | devices: {[d.device_type for d in tf.config.list_physical_devices()]}')
print(f'NumPy {np.__version__} | pandas {pd.__version__}')

# %%
# One quiet chart style for the whole notebook: hairline grid, left-aligned bold titles,
# one accent colour per series, one sequential ramp for matrices.
INK, INK_2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
BLUE, ORANGE, AQUA, RED, GOLD = '#2a78d6', '#eb6834', '#1baf7a', '#e34948', '#eda100'
BLUE_RAMP = LinearSegmentedColormap.from_list(
    'blue_ramp', ['#f0f6fe', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'figure.dpi': 110, 'font.size': 10, 'text.color': INK,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.labelcolor': INK_2,
    'axes.titlecolor': INK, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'axes.titlepad': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK_2, 'ytick.labelcolor': INK_2,
    'lines.linewidth': 2, 'lines.solid_capstyle': 'round',
    'legend.frameon': False, 'legend.fontsize': 9,
    'axes.prop_cycle': plt.cycler(color=[BLUE, ORANGE, AQUA, GOLD, '#e87ba4', '#008300', '#4a3aa7', RED]),
})


def barh(ax, labels, values, color=BLUE, title=None, xlabel=None, fmt='{:,.0f}'):
    """Horizontal bar chart, largest bar on top, value labels at the bar ends."""
    values = np.asarray(values, dtype=float)
    pos = np.arange(len(labels))
    ax.barh(pos, values, height=0.64, color=color)
    ax.set_yticks(pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    ax.tick_params(axis='y', length=0)
    span = values.max() if len(values) else 1.0
    for p, v in zip(pos, values):
        ax.text(v + span * 0.012, p, fmt.format(v), va='center', ha='left', fontsize=9, color=INK_2)
    ax.set_xlim(0, span * 1.14)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    return ax


def heat(ax, M, xlabels, ylabels, title=None, fmt='{:.0f}', cmap=BLUE_RAMP, vmax=None, vmin=0):
    """Annotated matrix with readable text contrast on dark cells."""
    M = np.asarray(M, dtype=float)
    vmax = float(np.nanmax(M)) if vmax is None else vmax
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.grid(False)
    ax.tick_params(length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=8,
                    color='white' if v > vmax * 0.55 else INK_2)
    if title:
        ax.set_title(title)
    return ax


def wrap(text, width=94, indent=''):
    """Print long generated text as a readable block."""
    for para in str(text).split('\n'):
        print(textwrap.fill(para, width=width, initial_indent=indent, subsequent_indent=indent)
              if para.strip() else '')
