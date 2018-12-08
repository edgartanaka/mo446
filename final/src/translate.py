import pandas as pd
import mat4py

# translate sorted_target_data.mat to CSV
df = mat4py.loadmat('../matlab/data/Experiment data/sorted_target_data.mat')
features_list = [x[0] for x in df['sorted_target_data']]
df = pd.DataFrame.from_dict(features_list)
df.to_csv('sorted_target_data.csv')

df = mat4py.loadmat('../matlab/data/Experiment data/sorted_target_data.mat')
features_list = [x[0] for x in df['sorted_target_data']]
df = pd.DataFrame.from_dict(features_list)
df.to_csv('sorted_target_data.csv')