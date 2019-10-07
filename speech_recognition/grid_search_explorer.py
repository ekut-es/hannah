GENERAL_EXCLUDE_FILE = "general_exclude.lst"
VALUE_FILE_SUFFIX = ".val"
GENERAL_SETTINGS_FILE = "general_settings.conf"
GENERAL_SETTINGS_FILE_DELIMITER = "="

import os
from .train import build_config
from .grid_search_explorer_modules.combinator import *
from .grid_search_explorer_modules.scheduler import *
import itertools
import time

def show_title():
    print(" ================================== ")
    print(" ====== Grid Search Explorer  ===== ")
    print(" ================================== ")
    print("")
    
def extract_value_by_key(lines, key):
    for line in lines:
        line = line.rstrip("\n")
        fields = line.split(GENERAL_SETTINGS_FILE_DELIMITER)
        if(fields[0] == key):
            return fields[1]
    raise Exception(f"Key {key} not found")
        

def open_config_file(config_file_path):
    config_file_existing = os.path.isfile(config_file_path)
    config_file_lines = []
    if(config_file_existing):
        with open(config_file_path, "r") as f:
            for line in f:
                config_file_lines.append(line)
    return (config_file_existing, config_file_lines)
    
def ask_yes_no(message, default_yes = True):
    if(default_yes):
        yes_no_str = "(Y/n)"
    else:
        yes_no_str = "(y/N)"
    while True:
        choice = input(f" {message} {yes_no_str}?: ")
        if(choice.lower() == "y"):
            return True
        elif(choice.lower() == "n"):
            return False
        elif(choice.lower() == ""):
            return default_yes
            
def ask_bool(default):
    result = ""
    if(default == True):
        result = input("0 / -> 1 <- ?: ")
    else:
        result = input("-> 0 <- / 1 ?: ")
    if(result == ""):
        if(default == True):
            return 1
        else:
            return 0
    else:
        return result
        
def ask_str(default):
    result = input(f"default={default} ?: ")
    if(result == ""):
        return default
    else:
        return result
        
def ask_int(message, default):
    result = input(f"{message}: default={default} ?: ")
    if(result == ""):
        return default
    else:
        return int(result)
        
def ask_float(message, default):
    result = input(f"{message}: default={default} ?: ")
    if(result == ""):
        return default
    else:
        return float(result)
        
def ask_list_of(datatype, values):
    last_input = ""
    iterations = 0
    return_values = []
    while True:
        if(iterations < len(values)):
            if(iterations > 0):
                last_input = input(f"Enter a value (or 'q' for quit) [default with ENTER={values[iterations]}]: ")
                if(last_input == "q"):
                    break
            else:
                last_input = input(f"Enter a value [default with ENTER={values[iterations]}]: ")
            if(last_input == ""):
                return_values += [values[iterations]]
                iterations += 1
                continue
        else:
            if(iterations > 0):
                last_input = input(f"Enter a value (or 'q' for quit) [default with ENTER=q]: ")
                if(last_input == "q" or last_input == ""):
                    break
            else:
                last_input = input(f"Enter a value: ")
        try:
            return_values += [datatype(last_input)]
        except ValueError:
            raise Exception("Couldnt parse value!")
        iterations += 1
    return return_values
                 
def ask_values_again(modelname, key, default):
    entrytype, entries = load_csv(modelname, key, payload=-1)
    result = None
    if(isinstance(default, bool)):
        result = f"{key};bool;{str(int(ask_bool(default=entries[0])))}"
    elif(isinstance(default, str)):
        result = f"{key};str;{ask_str(default=entries[0])}"
    elif(isinstance(default, int)):
        want_start_stop_step = ask_yes_no("Do you want start-stop-step (alternative: predefined)", default_yes = (entrytype == "int"))
        if(want_start_stop_step != (entrytype == "int")):
            return ask_values_first_time(modelname = modelname, key = key, default = default, skip_ask_whether_predefined = True, skip_ask_want_start_stop_step = want_start_stop_step)
        if(want_start_stop_step):
            start = ask_int("Start", int(entries[0]))
            stop = ask_int("Stop", int(entries[1]))
            steps = ask_int("Steps", int(entries[2]))
            result = f"{key};int;{start};{stop};{steps}"
        else:
            results = ask_list_of(int, values=entries)
            result = f"{key};predefined_int"
            for value in results:
                result += f";{str(value)}"
    elif(isinstance(default, float)):
        want_start_stop_step = ask_yes_no("Do you want start-stop-step (alternative: predefined)", default_yes = (entrytype == "float"))
        if(want_start_stop_step != (entrytype == "float")):
            return ask_values_first_time(modelname = modelname, key = key, default = default, skip_ask_whether_predefined = True, skip_ask_want_start_stop_step = want_start_stop_step)
        if(want_start_stop_step):
            start = ask_float("Start", float(entries[0]))
            stop = ask_float("Stop", float(entries[1]))
            steps = ask_int("Steps", int(entries[2]))
            result = f"{key};float;{start};{stop};{steps}"
        else:
            results = ask_list_of(float, values=entries)
            result = f"{key};predefined_float"
            for value in results:
                result += f";{str(value)}"
    elif(isinstance(default, list)):
        result = f"{key};list"
        for i in range(0, len(entries) // 3):
            print(f"List position Nr. {i}:")
            if(isinstance(default[0], int)):
                start = ask_int("Start",int(entries[i * 3]))
                stop = ask_int("Start",int(entries[i * 3 + 1]))
                steps = ask_int("Start",int(entries[i * 3 + 2]))
            elif(isinstance(default[0], float)):
                start = ask_float("Start",float(entries[i * 3]))
                stop = ask_float("Start",float(entries[i * 3 + 1]))
                steps = ask_int("Start",int(entries[i * 3 + 2]))
            else:
                raise Exception("List element type unsupported")
            result += f";{start};{stop};{steps}"
    assert result != None
    return result 

def ask_values_first_time(modelname, key, default, skip_ask_whether_predefined = False, skip_ask_want_start_stop_step = True):
    result = None
    if(isinstance(default, bool)):
        result = f"{key};bool;{str(int(ask_bool(default=default)))}"
    elif(isinstance(default, str)):
        path = os.path.join(CLASSES_DIRECTORY, key + CLASS_EXTENSION)
        presetting = ""
        if(os.path.isdir(path)):
            entries = [x for x in sorted(os.listdir(path))]
            presetting = entries[0]
        elif(os.path.isfile(path)):
            entries = []
            with open(path, "r") as f:
                entries = [x.rstrip("\n") for x in f]
            presetting = entries[0]
        result = f"{key};str;{ask_str(default=presetting)}"
    elif(isinstance(default, int)):
        want_start_stop_step = skip_ask_want_start_stop_step
        if(not skip_ask_whether_predefined):
            want_start_stop_step = ask_yes_no("Do you want start-stop-step (alternative: predefined)", default_yes = True)
        if(want_start_stop_step):
            start = ask_int("Start", default)
            stop = ask_int("Stop", default)
            steps = ask_int("Steps", 0)
            result = f"{key};int;{start};{stop};{steps}"
        else:
            results = ask_list_of(int, values=[])
            result = f"{key};predefined_int"
            for value in results:
                result += f";{str(value)}"          
    elif(isinstance(default, float)):
        if(not skip_ask_whether_predefined):
            want_start_stop_step = ask_yes_no("Do you want start-stop-step (alternative: predefined)", default_yes = True)
        if(want_start_stop_step):
            start = ask_float("Start", default)
            stop = ask_float("Stop", default)
            steps = ask_int("Steps", 0)
            result = f"{key};float;{start};{stop};{steps}"
        else:
            results = ask_list_of(float, values=[])
            result = f"{key};predefined_float"
            for value in results:
                result += f";{str(value)}"
    elif(isinstance(default, list)):
        result = f"{key};list"
        for i in range(len(default)):
            print(f"List position Nr. {i}:")
            if(isinstance(default[0], int)):
                start = ask_int("Start", default[i])
                stop = ask_int("Stop", default[i])
                steps = ask_int("Steps", 0)
            elif(isinstance(default[0], float)):
                start = ask_float("Start", default[i])
                stop = ask_float("Stop", default[i])
                steps = ask_int("Steps", 0)
            else:
                raise Exception("List element type unsupported")   
            result += f";{start};{stop};{steps}"
    assert result != None
    return result   

def main():
    model_name, config = build_config()

    show_title()
    
    print("Checking for existing general exclude file...")
    
    general_exclude_file_path = os.path.join(CONF_DIR, GENERAL_EXCLUDE_FILE)
    
    general_exclude_file_existing, general_exclude_file_lines = open_config_file(general_exclude_file_path)
    
    assert general_exclude_file_existing
    
    print("... FOUND!")
    
    print("Checking for existing exclude file...")
    
    exclude_file_path = os.path.join(CONF_DIR, model_name + EXCLUDE_FILE_SUFFIX)
    
    exclude_file_existing, exclude_file_lines = open_config_file(exclude_file_path)
    
    shall_skip = False

    exclude_file_lines_to_write = exclude_file_lines
    
    if(exclude_file_existing):
        print("... FOUND!")
        shall_skip = ask_yes_no("Skip the hyperparameter selection", default_yes = True)
    else:
        print("... NOT FOUND!")
        
    if(not shall_skip):
    
        exclude_file_lines_to_write = []
        
        print(f" You have to choose from {len(config.items()) - len(general_exclude_file_lines)} possible hyperparameters to be variable...")
        
        for key, value in sorted(config.items()):
            if(not key + "\n" in general_exclude_file_lines):
                default_exclude = (key + "\n" in exclude_file_lines)
                shall_exclude = (not ask_yes_no(f"-> Include key={key}, defaultvalue={value} ", default_yes = (not default_exclude)))
                if(shall_exclude):
                    exclude_file_lines_to_write += [key + "\n"]
                    
    print("(Re-)Writing exclude file...")
                    
    with open(exclude_file_path, "w") as f:
        for line in exclude_file_lines_to_write:
            f.write(line)                
    
    exclude_file_lines = exclude_file_lines_to_write                 
                    
    print("Checking for existing value file...")
    value_file_path = os.path.join(CONF_DIR, model_name + VALUE_FILE_SUFFIX)
    
    value_file_existing, value_file_lines = open_config_file(value_file_path)
    
    shall_skip = False
    allow_skip = True
    
    if(value_file_existing):
        print("... FOUND!")
        print("Checking if all settings are available...")
        for key, value in config.items():
            if(not key + "\n" in general_exclude_file_lines):
                found_key = False
                for value_line in value_file_lines:
                    value_key = value_line.split(VALUE_FILE_DELIMITER)[0]
                    if key == value_key:
                        found_key = True
                        break
                if(not found_key):
                    print(f"Setting for key: {key} is missing!") 
                    allow_skip = False
        if(allow_skip):
            print("... all settings are configured!")               
            shall_skip = ask_yes_no("Skip value settings", default_yes = True)
    else:
        print("... NOT FOUND!")
    
    if(not shall_skip):
        lines_to_write = []
        for key, value in sorted(config.items()):
            if(not key + "\n" in general_exclude_file_lines):
                found_key = False
                for value_line in value_file_lines:
                    value_key = value_line.split(VALUE_FILE_DELIMITER)[0]
                    if key == value_key:
                        found_key = True
                        break
                result = None
                if(not found_key):
                    if(isinstance(value, bool) or isinstance(value, int) or isinstance(value, float) or isinstance(value, str) or isinstance(value, list)):
                        print(f"Setting (combination) values for key: {key} the first time:") 
                        result = ask_values_first_time(model_name, key, value)
                    else:
                        print(f"Skipping {key} due to unsupported datatype...")
                else:
                    print(f"Setting (combination) values for key: {key} again:")
                    result = ask_values_again(model_name, key, value)
                lines_to_write += [result + "\n"]
                
        print("Writing value file...")
        with open(value_file_path, "w") as f:
            for line in lines_to_write:
                f.write(line)
        print("... Done!")
        
    print("Building the cartesian product of all settings...")
    variants = variants_from_valuefile(model_name, value_file_path)
            
            
    print("Reading general settings file...")
    gsettings_file_path = os.path.join(CONF_DIR, GENERAL_SETTINGS_FILE)
    
    gsettings_file_existing, gsettings_file_lines = open_config_file(gsettings_file_path)
    
    assert gsettings_file_existing == True
    
    gpus_string = ""
    
    while gpus_string == "":
        gpus_string = input("\nPlease enter the indices of gpus wanted for training (comma-separated) : ")
        
    gpus = [x for x in map(int, gpus_string.split(","))]
    
    print("Now starting scheduler...")
    
    gsettings = dict()
    
    for line in gsettings_file_lines:
        splitted_line = line.split(GENERAL_SETTINGS_FILE_DELIMITER)
        gsettings[splitted_line[0]] = splitted_line[1].rstrip("\n")
        
    scheduler = Scheduler()
    
    scheduler.set_model_name(model_name)
    scheduler.set_allowed_gpus(gpus)
    scheduler.set_general_settings(gsettings)
    
    for variant in variants:
        scheduler.add_job_to_queue(variant)
        
    while(not scheduler.has_finished()):
        scheduler.print_status()
        scheduler.schedule()
        time.sleep(1)
    
    
if __name__ == "__main__":
    main()
