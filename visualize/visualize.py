import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import json
import argparse

CSV_DELIMITER = ","
TRAINED_MODELS_DIR = "trained_models"
CONFIGS_DIR = "configs"
EVAL_CSV_NAME = "eval.csv"
CONFIG_JSON_NAME = "config.json"
GENERAL_EXCLUDE_FILE = "general_exclude.lst"
GENERAL_EXCLUDE_DIR = "grid_search/config"
CLASSES_DIR = "grid_search/config/classes"
CLASSES_EXTENSION = ".opt"

def find_index_mapping_for_string_key(key, value):
    for file_or_subdir in sorted(os.listdir(CLASSES_DIR)):
        if(file_or_subdir == key + CLASSES_EXTENSION):
            path = os.path.join(CLASSES_DIR, file_or_subdir)
            if(os.path.isfile(path)):
                with open(path, "r") as f:
                    iteratelist = [(x, y.rstrip("\n")) for x, y in enumerate(sorted(f))]
                    for index, entry in iteratelist:
                        if(entry == value):
                            return (index, iteratelist)
            elif(os.path.isdir(path)):
                iteratelist = [x for x in enumerate(sorted(os.listdir(path)))]
                for index, subpath in iteratelist:
                    if(subpath == value):
                        return (index, iteratelist)         
    raise Exception(f"key={key} with value={value} not found!")
                    

parser = argparse.ArgumentParser()

parser.add_argument('--model')
parser.add_argument('--experiment_id')

args = parser.parse_args()

experiment_id = args.experiment_id
model_name = args.model

general_exclude_file_path = os.path.join(GENERAL_EXCLUDE_DIR, GENERAL_EXCLUDE_FILE)

general_excludes = []

with open(general_exclude_file_path, "r") as f:
    for line in f:
        line = line.rstrip("\n")
        general_excludes += [line]

eval_csv_path = os.path.join(TRAINED_MODELS_DIR, experiment_id, model_name, EVAL_CSV_NAME)

dataframe = list()

with open(eval_csv_path, "r") as f:
    is_first_line = True
    for line_csv in f:
        if(is_first_line):
            is_first_line = False
            continue
        line_csv = line_csv.rstrip("\n")
        hashval, phase, epoch, accuracy, loss, macs, weights, lr = line_csv.split(CSV_DELIMITER)
        config_json_path = os.path.join(TRAINED_MODELS_DIR, experiment_id, model_name, CONFIGS_DIR, hashval, CONFIG_JSON_NAME)
        with open(config_json_path, "r") as g:
            config_unfiltered = json.loads(g.read())
            config = dict()
            for key,value in config_unfiltered.items():
                if(not key in general_excludes):
                    if(isinstance(value, list)):
                        for position, element in enumerate(value):
                            config[key + f"*{position}"] = float(element)
                    elif(isinstance(value, str)):
                        config[key] = value
                    else:        
                        config[key] = float(value)
                config["accuracy"] = float(accuracy)
            dataframe += [config]
                

pd_dataframe = pd.DataFrame(dataframe)
# From https://stackoverflow.com/questions/39658574/how-to-drop-columns-which-have-same-values-in-all-rows-via-pandas-or-spark-dataf
nunique = pd_dataframe.apply(pd.Series.nunique)
cols_to_drop = nunique[nunique == 1].index
pd_dataframe = pd_dataframe.drop(cols_to_drop, axis=1)

print(pd_dataframe)

dimensions = list()


for column in pd_dataframe:
    dimension = dict()
    value = pd_dataframe[column][0]
    if(isinstance(value, str)):
        _, iteratelist = find_index_mapping_for_string_key(key=column, value=value)
        valrange = [0, len(iteratelist) - 1]
        tickvals = [x for x in range(len(iteratelist))]
        ticktext = [x for _, x in iteratelist]
        values = list()
        for value_dataframe in pd_dataframe[column]:
            for index, value in iteratelist:
                if(value == value_dataframe):
                    values += [index]
        dimension = dict(range=valrange, label=column, values=values, tickvals = tickvals, ticktext = ticktext)
            
    else:
        dimension = dict(label=column, values=pd_dataframe[column])
    dimensions += [dimension]


fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = pd_dataframe['accuracy'],
                   colorscale = 'Electric',
                   showscale = True,
                   cmin = 0,
                   cmax = 100),
        dimensions=dimensions
    )
)
fig.show()
