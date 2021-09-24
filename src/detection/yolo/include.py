# import
import os
import sys
import shutil
import yaml
import json
from collections import defaultdict
from glob import glob
from tqdm.notebook import tqdm
import gc
import re
import argparse
import kaggle
import zipfile
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import torch

# global variables
WORKING_BASE = os.getcwd()
INPUT_BASE = os.path.join(WORKING_BASE, 'kaggle_datasets')
DATA_BASE = os.path.join(WORKING_BASE, 'kaggle_datasets')

# configs
os.environ['WANDB_MODE'] = 'dryrun'
#sys.path.extend([os.path.join(os.getcwd(), 'hung_repo', 'dummy0'),
#                 os.path.join(os.getcwd(), 'hung_repo', 'dummy_siim')])

from hung_common import *
from hung_torch_common import *
from him import *
from file import *
from siim_yolo import *
from siim_map import *
