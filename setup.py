# Commonly used built-in imports
import datetime as dt
from glob import glob
import itertools as it
import os
import sys
import warnings

# Commonly used external imports
import geopandas as gpd
from geopandas import GeoDataFrame as Gdf
from igraph import Graph
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import array as Arr
import pandas as pd
from pandas import DataFrame as Pdf
from pandas import Series as Seq
import pyspark
from pyspark.sql import functions as F
import seaborn as sns
import shapely
from tqdm.notebook import tqdm

# Imports from mobilkit
sys.path.insert(0, '/home/umni2/a/umnilab/apps/mobilkit')
import mobilkit as mk
from mobilkit import utils as U
from mobilkit.spark import Types as T
from mobilkit.geo import CRS_DEG, CRS_M

# Display settings
mk.utils.config_display(disp_method=True)
plt.rcParams.update(mk.utils.MPL_RCPARAMS)
mpl.rcParams.update(mk.utils.MPL_RCPARAMS)

# Pyspark session handler
# configuration depends on the server
# Server on which the target script will run
_MACHINE = {
    'tnet1.ecn.purdue.edu': 'tnet',
    'umni5.ecn.purdue.edu': 'umni5',
    'umni2.ecn.purdue.edu': 'umni2',
}[os.uname().nodename]
# Pyspark session handler with resources allocated 
# according to the host server
SP = mk.spark.Spark({
    'executor.memory': dict(umni5='160g', tnet='200g', umni2='36g')[_MACHINE],
    'driver.memory': dict(umni5='160g', tnet='200g', umni2='36g')[_MACHINE],
    'default.parallelism': dict(umni5=32, tnet=16, umni2=20)[_MACHINE]
}, start=False)

# Important paths
class IO:
    # common root directory of TNET-1, UMNI-2 & UMNI-5
    root = '/home/umni2/a/umnilab/'
    # root directory for all my projects
    home = root + 'users/verma99/'
    # base folder containing cleaned Quadrant ping data
    quad = root + 'data/Quadrant/'
    # base folder containing SafeGraph's POI data
    sg = root + 'data/SafeGraph/'
    # project's home directory
    proj = home + 'pednet/'
    # folder for all the datasets specific to this project
    data = U.mkdir(proj + 'data/')
    # folder for all the images specific to this project
    fig = U.mkdir(proj + 'figures/')


class City:
    def __init__(self, geocode, root, name=None, load=()):
        self.geocode = geocode
        self.name = name if isinstance(name, str) else geocode.split(',')[0]
        self.label = self.name.lower().replace(' ', '_')
        self.root = U.mkdir(f'{root}/{self.label}/')
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


class Pednet:
    def __init__(self, city, query=None, name='', od_fname='', E=None):
        # load prepared data for this city
        V = city.load('full_pednet_nodes')
        if E is None:
            E = city.load('simple_sidewalk_pednet_edges')
        odpt2V = city.load('odpt2node').set_index('odpt_id')
        od = pd.read_pickle(city.root + od_fname + '.pickle')
        self.city = city
        self.name = name
        E = E.query(query) if isinstance(query, str) else E
        V = V.loc[list(set(E['src_vid']) | set(E['trg_vid']))].sort_index()
        G = Graph()
        G.add_vertices(V.shape[0], {'vid': V.index})
        vid2idx = pd.Series(range(V.shape[0]), index=V.index)
        end_pts = list(zip(vid2idx.loc[E['src_vid']], vid2idx.loc[E['trg_vid']]))
        E2 = E.rename_axis('id').drop(columns=['id','geometry'],
                                      errors='ignore').reset_index()
        G.add_edges(end_pts, {x: E2[x].values for x in E2.columns})
        V['cid'] = G.clusters().membership
        E = E.merge(V['cid'].rename('src_cid'), left_on='src_vid',
                    right_index=True, how='left')
        E = E.merge(V['cid'].rename('trg_cid'), left_on='trg_vid',
                    right_index=True, how='left')
        E['cid'] = E['src_cid'] | E['trg_cid']
        E = E.drop(columns=['src_cid','trg_cid']).rename_axis('id')
        
        odpt2V['dist'] = odpt2V['dist_odpt2cp'] + odpt2V['dist_cp2node']
        od = (od.drop_duplicates().reset_index()
              .merge(od.groupby(['orig_odpt','dest_odpt']).size()
                     .rename('n_ods'), on=('orig_odpt','dest_odpt'))
              .merge(odpt2V[['vid','dist']].rename(columns=lambda x: x + '_o'),
                     left_on='orig_odpt', right_index=True)
              .merge(odpt2V[['vid','dist']].rename(columns=lambda x: x + '_d'),
                     left_on='dest_odpt', right_index=True)
              .assign(dist = lambda df: df.pop('dist_o') + df.pop('dist_d'))
              .astype({'vid_o': np.int32, 'vid_d': np.int32})
              .set_index('od_id').sort_index())
        self.V, self.E, self.G, self.vid2idx, self.od = V, E, G, vid2idx, od
    
    def __repr__(self):
        return f'Pednet("{self.name}" in {self.city})'
    
    def get_sp(self, o, d):
        oid, did = self.vid2idx.loc[o], self.vid2idx.loc[d]
        sp = self.G.get_shortest_paths(oid, did, 'len', output='epath')[0]
        edges = [e['id'] for i, e in enumerate(self.G.es) if i in sp]
        return {'vid_o': o, 'vid_d': d, 'edges': edges,
                'd_V2V': self.E.loc[edges]['len'].sum()}
