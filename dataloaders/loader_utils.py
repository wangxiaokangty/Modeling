import os
import pandas as pd
import random
from dataloaders.valid_data_fields import *
import numpy as np
import logging

'''
data loading utilities, for the most part these guys deal with extracting information from the csvs as well as dealing with some 
'qirks' in the data
'''

data_base_path= 'data/'
def get_final_data(open_years, get_match_info=False, normalize=False):
    data = []
    print(open_years)
    for open_year in open_years:
        parsed_csv_path = os.path.join(data_base_path, 'final_data', f'{open_year}-points-final.csv')
        data += get_training_data_from_parsed_csvs(parsed_csv_path, get_match_info)
    if normalize:
        data = normalize_data(data)
    return data

def get_training_data_from_parsed_csvs(parsed_points_path, get_match_info=False):
    all_points = pd.read_csv(parsed_points_path) 

    good_matches = all_points.loc[all_points['winner'] != 0]['match_id'].unique()
    data = []
    dropped_matches = 0

    for match_id in good_matches:
        try:
            match_data = all_points.loc[all_points['match_id'] == match_id]
            
            t_data, label = get_parsed_match_data(match_id, match_data)
            if get_match_info:
                p1 = match_data['player1'].iloc[0]
                p2 = match_data['player2'].iloc[0]
                winner = match_data['winner'].iloc[0]
                data.append([t_data, label, f'{p1} vs {p2} winner was {winner}, {match_id}'])
            else:
                data.append([t_data, label])
        except Exception as e:
            #logging.exception(e)
            print(match_id)
            dropped_matches += 1
    print(f'dropped {dropped_matches} matches')
    return data


def get_parsed_match_data(match_id, point_data):
    
    winner = point_data.iloc[0]['winner'] - 1

    parsed_point_data = extract_numpy_from_parsed_match(point_data)
    num_points = parsed_point_data.shape[0]

    assert num_points > 0, "dropped match due to no points played"
    y_gt = np.full(shape=num_points, fill_value=winner, dtype='float')
    return parsed_point_data, y_gt

def extract_numpy_from_parsed_match(match_points):
    
    match_points_copy = match_points.copy()
    parsed_time = match_points_copy
    # parsed_time['ElapsedTime'] = parsed_time['ElapsedTime'].map(lambda x: parse_time(x))
    parsed_scores = parsed_time.replace('AD', 55)
    
    scores = parsed_scores[valid_fields].to_numpy(dtype=np.float64)
    # scores = scores.fillna(0)
    assert np.sum(np.isnan(scores)) < 1, f"hit a nan {scores}"
    # scores = scores[~np.isnan(scores).any(axis=1)]
    return scores
