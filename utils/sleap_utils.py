import sys

import numpy as np
import pandas as pd
import h5py

sys.path.append(r"D:\codes\python\courtship\utils")
from utils import *



def readTracks(h5_path: str) -> list:
    """
    read a sleap h5 and save each track to a pd.DataFrame. 
    """

    track_dfs = []

    with h5py.File(h5_path, 'r') as f:
        instance_scores = f['instance_scores'][:]
        node_names = f['node_names'][:]
        point_scores = f['point_scores'][:]
        tracking_scores = f['tracking_scores'][:]
        occupancy = f['track_occupancy'][:].transpose()
        tracks_mat = np.transpose(f['tracks'][:], axes=[0,2,1,3])
    
    for (instance_score, point_score, tracking_score, occu, track_mat) in zip(
        instance_scores, point_scores, tracking_scores, occupancy, tracks_mat):
        
        track_df = pd.DataFrame({
            'instance_score': instance_score, 
            'tracking_score': tracking_score, 
            'occupancy': occu, 
        })

        for idx_name, node_name in enumerate(node_names):
            node_name = node_name.astype(str)
            track_df['point_score_'+node_name] = point_score[idx_name,:]
            track_df['x_'+node_name] = track_mat[idx_name, 0, :]
            track_df['y_'+node_name] = track_mat[idx_name, 1, :]
            track_df['displacement_'+node_name] = np.sqrt(
                track_df['x_'+node_name].diff().values**2 + 
                track_df['y_'+node_name].diff().values**2)

        track_df['frame'] = track_df.index
        
        track_dfs.append(track_df)        

    return track_dfs




def addMetrics(track_df: pd.DataFrame, max_displacement: float, 
    min_instance_score: float = 0.8, min_point_score: float = 0.8) -> pd.DataFrame:
    """
    add a column ('good') of bool type. 
    """

    point_col_names = filter(lambda s: (s.startswith('x_') or s.startswith('y_')), track_df.columns)
    displacement_col_names = filter(lambda s: (s.startswith('displacement_')), track_df.columns)

    occu_bool = track_df['occupancy']==1
    instance_score_bool = track_df['instance_score'] > min_instance_score

    point_score_bool = np.repeat(True, len(track_df))
    for point_col_name in point_col_names:
        point_score_bool = point_score_bool & (track_df[point_col_name] > min_point_score)
    
    displacement_bool = np.repeat(True, len(track_df))
    for displacement_col_name in displacement_col_names:
        displacement_bool = displacement_bool & (track_df[displacement_col_name] < max_displacement)

    good_bool = occu_bool & instance_score_bool & point_score_bool & displacement_bool

    track_df['good'] = good_bool
    
    return track_df




def get_flyclips(track_df: pd.DataFrame, min_consecutive: int) -> list:
    """
    get consecutive good frames and slice them out. 
    remove useless columns. 
    """
    
    flyclips = []
    for start, end in find_consecutive(track_df['good'], min_consec=min_consecutive):
        flyclips.append(track_df.iloc[start:end])

    return flyclips



def get_pairing_info(h5_path: str):
    
    with h5py.File(h5_path, 'r') as f:
        occupancy = f['track_occupancy'][:].transpose()
    
    pairing = list(filter(lambda pair: pair[0] < pair[1], zip(*np.where(np.corrcoef(occupancy) > 0.5))))

    return pairing



def pair_tracks(track_dfs: list) -> pd.DataFrame:

    paired = pd.DataFrame()

    good = []

    for i_track, track_df in enumerate(track_dfs):

        paired['frame'] = track_df['frame']

        point_col_names = filter(lambda s: (s.startswith('x_') or s.startswith('y_')), track_df.columns)
        # displacement_col_names = filter(lambda s: (s.startswith('displacement_')), track_df.columns)

        for pcn in point_col_names: 
            paired[pcn + '_' +str(i_track)] = track_df[pcn]

        # for dcn in displacement_col_names:
        #     paired[dcn + '_' +str(i_track)] = track_df[dcn]

        paired['good_' + str(i_track)] = track_df['good']
        good.append(track_df['good'])

    paired['good'] = np.array(good).sum(axis=0) == 2

    return paired

    
