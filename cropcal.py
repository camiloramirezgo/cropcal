# -*- coding: utf-8 -*-
"""
Cropland calibration algorithm

The code in this script provides functions to calibrate cropland GIS data. 
It should be used through a Jupyter Notebook template that guides the user through the required calibration process

@author: Camilo Ramirez Gomez
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time, sys
from IPython.display import clear_output, display, Markdown
import os

def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

def update_progress(progress):
    '''
    Updates the progress bar of the calibration process and provides information of the status
    '''
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

def load_data(path, names = None):
    '''
    Reads a csv file containing all GIS data needed for the cropland calibration process
    '''
    df = pd.read_csv(path, names = names, low_memory=False)
    return df

def clean_data(df, crop_name):
    '''
    Cleans the dataframe, removing the row that do not contain cropland
    '''
    df.drop(df.loc[df[crop_name]<=0].index, inplace=True)
    
def convert_units(units, df, cell_size, cropland_name):
    '''
    Call the right function based on the selected units of convertion
    '''
    global cell_area
    cell_area = 0
    if units == 'hectares':
        cell_area = crop_to_ha(df, cell_size, cropland_name)
    elif units == 'dunams':
        cell_area = crop_to_dunam(df, cell_size, cropland_name)
    else:
        print('The units provided are not supported yet.')
    return cell_area

def crop_to_ha(df, cell_size, crop_name):
    '''
    Converts area data from cropland units to hectares
    '''
    df[crop_name] *= (cell_size**2) / 10000
    cell_area = (cell_size**2) / 10000
    return cell_area
    
def crop_to_dunam(df, cell_size, crop_name):
    '''
    Converts area data from cropland units to dunam
    '''
    df[crop_name] *= (cell_size**2) / 1000
    cell_area = (cell_size**2) / 1000
    return cell_area

def total_crop(df, crop_name):
    '''
    Returns te total crop area
    '''
    return df[crop_name].sum()


def cropland_per_admin(df, admin_name, crop_name, calib_data, calibration_value_name):
    '''
    Returns a dataframe containing the cropland area sum in each province or selected administrative area
    '''
    temp_df = df[[admin_name,crop_name]].groupby(admin_name).agg({crop_name:'sum'})
    temp_df[calibration_value_name] = temp_df.index.map(calib_data.groupby(admin_name)[calibration_value_name].sum())
    temp_df['Difference'] =(temp_df[crop_name] - temp_df[calibration_value_name])/temp_df[calibration_value_name]
    temp_df['Difference'].fillna(0, inplace=True)
    return temp_df


def AHP(data):
    '''
    Computes a Analytical Hierarchy Process (AHP):
    Takes an array representing half of a pairwise rank matrix, which compares
    the importance of decision parameters relatively between them. Afterwards,
    it calculates the other half of the matrix and the weighted value
    for each parameter, returning the weights and the Coherence Rate (CR)
    '''
    n = len(data)
    for i in range(n):
        for j in range(n-1-i):
            data[i].append(1/data[j+i+1][i])
    m = []
    for value in data:
        m.append(np.array(value))
    m = np.array(m)
    m_norm = m/m.sum(axis=0)
    weights = m_norm.mean(axis=1)

    w_values = m * weights
    w_values_sum = w_values.sum(axis=1)
    s_w = w_values_sum/weights
    CI = (s_w.mean()-n)/(n-1)
    RI_values = {2:0,3:0.58,4:0.9,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.51}
    RI = RI_values[n]
    CR = CI/RI
    return weights, CR

def get_matrix(criteria_dic):
    '''
    Calculates the pairwise comparison matrix for the AHP method
    '''
    list_scores = []
    list_criteria = []
    for criteria, scores in criteria_dic.items():
        list_scores.append(scores)
        list_criteria.append(criteria)

    return np.array(list_scores), list_criteria

def consistency(CR):
    '''
    Computes the Consistency Ratio (CR) of the AHP process
    '''
    if CR < 0.1:
        return 'consistent'
    elif (CR - 0.1) > 0 and (CR - 0.1) < 0.1:
        return 'slightly inconsistent'
    else:
        return 'inconsistent'
        
def display_scores(w_categories, CR, criteria):
    '''
    Prints the Consistency Ratio (CR) and scores for each criteria of the AHP process
    '''
    display(Markdown('The computed **CR** for the provided decision matrix is: **' + str(round(CR,3)) + '**, thus the decision matrix is {}'.format(consistency(CR))))
    for i, score in enumerate(w_categories):
        display(Markdown('The score for criteria **{}** is: **{}%**'.format(criteria[i],round(score*100,2))))

def minimum_value(df, value, layer_name, number_of_layers = 1):
    '''
    Redefines the minimum distance of the data from cero to the value of the cell_size.
    This avoids erros of dividing by cero
    '''
    df.loc[df[layer_name]<cell_size,layer_name] = value


def normalize_data(df, criterias):
    '''
    Takes a dataframe and a parameter and adds a normalized version of it to the dataframe. Curently based on the max min method
    '''
    normalized = [criteria + 'Normalized' for criteria in criterias]
    df[normalized] = pd.DataFrame([[0]*len(normalized)])
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler = scaler.fit(df[criterias])
    df[normalized] = scaler.transform(df[criterias])

    
def find_threshold(df, province, regional_crops_total, accuracy, variable, admin_name, cropland_name, number, total_number):
    '''
    Finds the threshold value of a dataframe parameter, for it to have a total sum matching the calibration target within a tolerance
    '''
    vector = df[admin_name] == province[0]
    crop_sum = total_crop(df.loc[vector],cropland_name)
    threshold_vec = sorted(set(df.loc[vector,variable]), reverse=True)
    threshold = threshold_vec[0]
    j = 1
    # j = int(len(threshold_vec) / 2)
    while (abs((crop_sum-regional_crops_total)/regional_crops_total) > accuracy):
        if j % 100 == 0:
            update_progress(number / total_number)
            print('Finding threshold for {} '.format(admin_name) + str(province[0]))
            print('Testing Threshold {} of {}'.format(j,len(threshold_vec)))
            print('Cropland sum: ' + str(crop_sum))
            print('Cropland target: ' + str(regional_crops_total) + '\n')
        crop_sum = total_crop(df.loc[(df[variable]<threshold) & (vector)],cropland_name)
        if (crop_sum-regional_crops_total)/regional_crops_total > accuracy:
            threshold_vec = threshold_vec[j:]
            j = int(len(threshold_vec) / 2)
            threshold = threshold_vec[j]
            # j += 1
        elif (crop_sum-regional_crops_total)/regional_crops_total < (-accuracy):
            j = int(j / 2) - 1
            threshold = threshold_vec[j]
            
    
    df.loc[(df[variable]>threshold) & (vector),cropland_name] = 0
    crop_sum = total_crop(df.loc[vector],cropland_name)
    i = 1
    while (abs((crop_sum-regional_crops_total)/regional_crops_total) > accuracy):
        if i % 100 == 0:
            update_progress(number / total_number)
            print('Finding threshold for {} '.format(admin_name) + str(province[0]))
            print('Fine tunnin...')
            print('Iteration ' + str(i))
            print('Cropland sum: ' + str(crop_sum))
            print('Cropland target: ' + str(regional_crops_total) + '\n')
        i += 1
        df.loc[(df[variable]==threshold) & (vector), cropland_name] -= cell_area
        df.loc[(df[cropland_name]<0) & (vector), cropland_name] = 0
        crop_sum = total_crop(df.loc[vector],cropland_name)
    return threshold

def save_data(df, path, layers = 'All'):
    '''
    Reads a csv file containing all GIS data needed for the cropland calibration process
    '''
    if layers == 'All':
        df.to_csv(path, index=False)
    else:
        df[layers].to_csv(path, index=False)
        

def add_data(df, target_layer, source_layer, province, admin_name):
    '''
    For administrative areas were the total cropland area is lower than the calibration target and outside the selected tolerance,
    the method adds new cropland data taken from another GIS cropland layer
    '''
    df.loc[df[admin_name]==province, target_layer] = df.loc[df[admin_name]==province, source_layer]
        

def run_calibration(df, cropland_name, accuracy, calib_data, w_categories, cell_size, 
                    admin_name, calibration_value_name, criteria_vec, inverted_relation, 
                    restrictions, fill_negative = False):
    '''
    Runs the calibration process taking the calibration targets for each administrative area and the overall dataframe,
    then cuts cropland data based on a suitability map, until the calibration target within a defined tolerance is reached
    '''
    prov_data = cropland_per_admin(df, admin_name, cropland_name, calib_data, calibration_value_name)
    cero_values = prov_data.loc[prov_data[calibration_value_name] == 0]
    for province in cero_values.index:
        df.loc[df[admin_name] == province,cropland_name] = 0
    
    if fill_negative:
        prov_data = cropland_per_admin(df, admin_name, cropland_name, calib_data, calibration_value_name)
        prov_data = prov_data.loc[prov_data[calibration_value_name] != 0]
        negative_diff = prov_data.loc[prov_data['Difference'] < -accuracy]
        
        for province in negative_diff.index:
            add_data(df, cropland_name, fill_negative, province, admin_name)
    
    normalize_data(df,criteria_vec)
    criteria_vector = []
    for criteria in criteria_vec:        
        if inverted_relation[criteria].lower() == 'y':
            criteria_name = '{}NormalizedInv'.format(criteria)
            df[criteria_name] = 1 - df['{}Normalized'.format(criteria)]
        else:
            criteria_name = '{}Normalized'.format(criteria)
        
        criteria_vector.append(criteria_name)    
    
    df['UnsuitableAreas'] = (df[criteria_vector] * w_categories).sum(axis=1)
    for key, value in restrictions.items():
        df.loc[df[key] == value, 'UnsuitableAreas'] = 1
    
    prov_data = cropland_per_admin(df, admin_name, cropland_name, calib_data, calibration_value_name)
    prov_data = prov_data.loc[prov_data[calibration_value_name] != 0]
    positive_diff = prov_data.loc[prov_data['Difference'] > accuracy]
    threshold = {}
    for i, province in enumerate(positive_diff.iterrows()):
        regional_crops_total = province[1][calibration_value_name]
        threshold[province[0]] = find_threshold(df, province, 
                 regional_crops_total,accuracy, 'UnsuitableAreas', admin_name, cropland_name, i, positive_diff.shape[0])
    update_progress(1)

def get_statistics(df, admin_name, cropland_name, calib_data, calibration_value_name, units):
    '''
    Computes an prints in Jupyter Notebooks, basic statistics about the total cropland area and the calibration target
    '''
    total_area = total_crop(df, cropland_name)
    crops_total = calib_data[calibration_value_name].sum()
    display(Markdown('The current total cropland area is: **' + str(round(total_area, 2)) +
                     '** {}. This represents a **'.format(units) + 
                     str(round((total_area-crops_total) / crops_total * 100,1)) +
                     '%** difference against the regional statistics provided.'))
    return cropland_per_admin(df, admin_name, cropland_name, calib_data, calibration_value_name)