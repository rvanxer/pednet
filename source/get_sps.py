from igraph import Graph
from pqdm.processes import pqdm

from setup import *

class Pednet:
    def __init__(self, city, query=None, name='Full'):
        # load prepared data for this city
        V = city.load('full_pednet_nodes')
        E = city.load('simple_sidewalk_pednet_edges')
        odpt2V = city.load('odpt2node').set_index('odpt_id')
        od = pd.read_pickle(glob(city.root + 'sample_OD*')[0])
        
        self.city = city
        self.name = name
        E = E.query(query) if isinstance(query, str) else E
        V = V.loc[list(set(E['src_vid']) | set(E['trg_vid']))]
        G = igraph.Graph()
        G.add_vertices(V.shape[0], {'vid': V.index})
        vid2idx = pd.Series(range(V.shape[0]), index=V.index)
        G.add_edges(list(zip(vid2idx.loc[E['src_vid']], vid2idx.loc[E['trg_vid']])),
                    {'id': E.index.values} | {x: E[x].values for x in set(E.columns) - {'geometry'}})
        V['cid'] = G.clusters().membership
        E = E.merge(V['cid'].rename('src_cid'), left_on='src_vid', right_index=True, how='left')
        E = E.merge(V['cid'].rename('trg_cid'), left_on='trg_vid', right_index=True, how='left')
        E['cid'] = E['src_cid'] | E['trg_cid']
        E = E.drop(columns=['src_cid','trg_cid']).rename_axis('id')
        
        odpt2V['dist'] = odpt2V['dist_odpt2cp'] + odpt2V['dist_cp2node']
        od = (od.drop_duplicates().reset_index()
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
    
    def get_one_sp(self, o, d):
        o, d = self.vid2idx.loc[o], self.vid2idx.loc[d]
        sp = self.G.get_shortest_paths(o, d, 'len', output='epath')[0]
        edges = [e['id'] for i, e in enumerate(self.G.es) if i in sp]
        return {'vid_o': o, 'vid_d': d, 'edges': edges, 'd_V2V': self.E.loc[edges]['len'].sum()}
    
    def get_sps(self, n_jobs, save=True):
        # filter unique connected OD pairs
        od = (self.od.drop_duplicates(subset=['vid_o','vid_d'])
              .merge(self.V['cid'].rename('cid_o'), left_on='vid_o', right_index=True)
              .merge(self.V['cid'].rename('cid_d'), left_on='vid_d', right_index=True)
              .query('cid_o == cid_d').drop(columns=['cid_o','cid_d']))
        od = od.head(20)
        sp = Pdf(pqdm(zip([self]*od.shape[0], od['vid_o'], od['vid_d']), self.get_one_sp, n_jobs=n_jobs, total=od.shape[0]))
        sp.insert(0, 'od_id', od.index)
        if save:
            self.city.save(f'paths_{self}', sp)
        return sp
    
    
if __name__ == '__main__':
    c = City('Cambridge, MA')
    c.s0 = Pednet(c, 'exists', 'Current')
    c.s1 = Pednet(c, 'exists or ~is_xwalk', '+Sidewalks')
    c.s2 = Pednet(c, 'exists or is_xwalk', '+Crosswalks')
    c.s3 = Pednet(c, None, '+Both')
    c.scs = [c.s0, c.s1, c.s2, c.s3]
        
    print(c.s0.get_sps(n_jobs=30))
