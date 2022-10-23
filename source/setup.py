from datetime import date as Date
from datetime import datetime as dt
from glob import glob
import itertools
import os
from pathlib import Path
import sys
import warnings

import geopandas as gpd
from geopandas import GeoDataFrame as Gdf
import igraph
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import array as Arr
import pandas as pd
from pandas import DataFrame as Pdf
from pandas import Series as Seq
import seaborn as sns
from tqdm.notebook import tqdm

# Mobilkit setup
sys.path.insert(0, '/home/umni2/a/umnilab/apps/mobilkit/')
import mobilkit as mk
from mobilkit import utils as U
from mobilkit.spark import Types as T
from mobilkit.geo import CRS_M, CRS_DEG

# display settings
U.config_display(disp_method=True)
plt.rcParams.update(mk.utils.MPL_RCPARAMS)
mpl.rcParams.update(mk.utils.MPL_RCPARAMS)

# project folder
HOME = '/home/umni2/a/umnilab/users/verma99/pednet_study/'
# data folder
DATA_DIR = U.mkdir(HOME + 'data/')

class City:
    def __init__(self, geocode, name=None, root=DATA_DIR, load=(), crs=CRS_DEG):
        self.geocode = geocode
        self.name = name if isinstance(name, str) else geocode.split(',')[0]
        self.label = self.name.lower().replace(' ', '_')
        self.root = U.mkdir(f'{root}{self.label}/')
        layers = [load] if isinstance(load, str) else load
        self.loads(layers)
    
    def __repr__(self):
        return f'City({self.name})'
    
    def load(self, layer, crs=None):
        if hasattr(self, layer):
            return getattr(self, layer)
        df = pd.read_pickle(self.root + layer + '.pickle')
        if crs is not None:
            df = df.to_crs(crs)
        setattr(self, layer, df)
        return df
    
    def loads(self, layers):
        for layer in layers:
            self.load(layer)

    def save(self, fname, df):
        setattr(self, fname, df)
        df.to_pickle(self.root + fname + '.pickle')
