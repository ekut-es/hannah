import abc
import os
import itertools

CONF_DIR = "speech_recognition/grid_search_explorer_config"
VALUE_FILE_DELIMITER = ";"
VALUE_FILE_DIRECTORY = CONF_DIR
VALUE_FILE_EXTENSION = ".val"
CLASSES_DIRECTORY = "speech_recognition/grid_search_explorer_config/classes/"
CLASS_EXTENSION = ".opt"
EXCLUDE_FILE_SUFFIX = "_exclude.lst"

def load_csv(modelname, key, payload):
    filename = os.path.join(VALUE_FILE_DIRECTORY, modelname + VALUE_FILE_EXTENSION)
    with open(filename, "r") as f:
        for line in f:
            line = line.replace("\n", "")
            splitted_line = line.split(VALUE_FILE_DELIMITER)
            line_key = splitted_line[0]
            if(line_key == key):
                if(payload >= 0):
                    return (splitted_line[1], splitted_line[2:2+payload])
                else:
                    return (splitted_line[1], splitted_line[2:])
    raise Exception(f"ERROR: load_csv: key >>{key}<< not found in file {filename}")
    
def float_range(start, stop, steps):
    if(steps == 0):
        return [start]
    result = []
    accu = start
    increment = (stop - start) / steps
    for _ in range(steps + 1):
        result += [accu]
        accu += increment
    return result

class Combinator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate_variants(self):
        pass
        
class Decorator(Combinator, metaclass=abc.ABCMeta):

    _modelname = None
    _key = None
    
    def __init__(self, chain_object):
        self._chain_object = chain_object
        
    def set_key(self, key):
        self._key = key
        
    def set_modelname(self, modelname):
        self._modelname = modelname
        
    def factory(chain, modelname, key):
        
        result = None
        datatype, _ = load_csv(modelname, key, payload = 0)
        if(datatype == "bool"):
            result = CombinatorBoolean(chain)
        if(datatype == "int"):
            result = CombinatorInteger(chain)
        if(datatype == "float"):
            result = CombinatorFloat(chain)
        if(datatype == "predefined_int"):
            result = CombinatorPredefinedInteger(chain)
        if(datatype == "predefined_float"):
            result = CombinatorPredefinedFloat(chain)
        if(datatype == "str"):
            result = CombinatorString(chain) 
        if(datatype == "list"):
            result = CombinatorList(chain)           
        assert result != None
        result.set_modelname(modelname)
        result.set_key(key)
        return result
        
    def check_whether_excluded(self, modelname, key):
        exclude_file_path = os.path.join(CONF_DIR, modelname + EXCLUDE_FILE_SUFFIX)
        with open(exclude_file_path, "r") as f:
            return (key + "\n" in f)
        
class TailClass(Combinator):
    def generate_variants(self):
        return []
        
class CombinatorBoolean(Decorator):
    def generate_variants(self):
        value = self.load_csv_boolean()
        if self.check_whether_excluded(self._modelname, self._key):
            return self._chain_object.generate_variants() + [(self._key, [value])]
        else:
            return self._chain_object.generate_variants() + [(self._key, [value, not value])]
    def load_csv_boolean(self):
        _, entries = load_csv(self._modelname, self._key, payload=1)
        return bool(int(entries[0]))

class CombinatorInteger(Decorator):
    def generate_variants(self):
        start, stop, steps = self.load_csv_integer()
        if(steps == 0 or self.check_whether_excluded(self._modelname, self._key)):
            return self._chain_object.generate_variants() + [(self._key, [start])]
        else:
            return self._chain_object.generate_variants() + [(self._key, [x for x in range(start, stop + 1, (stop - start) // steps)])]
        
    def load_csv_integer(self):
        _, entries = load_csv(self._modelname, self._key, payload=3)
        return (int(entries[0]), int(entries[1]), int(entries[2]))
        
class CombinatorFloat(Decorator):
    def generate_variants(self):
        start, stop, steps = self.load_csv_float()
        if(self.check_whether_excluded(self._modelname, self._key)):
            return self._chain_object.generate_variants() + [(self._key, [start])]
        else:
            return self._chain_object.generate_variants() + [(self._key, float_range(start, stop, steps))]
    def load_csv_float(self):
        _, entries = load_csv(self._modelname, self._key, payload=3)
        return (float(entries[0]), float(entries[1]), int(entries[2]))
        
class CombinatorList(Decorator):
    def generate_variants(self):
        entries = self.load_csv_list()
        assert len(entries) % 3 == 0
        combinator_list = []
        conversion_worked = True
        try:
            entries[0] = int(entries[0])
        except ValueError:
            conversion_worked = False
            
        if(not conversion_worked):
            entries[0] = float(entries[0])
            
        for i in range(0, len(entries) // 3): 
            if(isinstance(entries[0], int)): 
                start = int(entries[i * 3])
                stop = int(entries[i * 3 + 1])
                steps = int(entries[i * 3 + 2])
                if(steps == 0 or self.check_whether_excluded(self._modelname, self._key)):
                    combinator_list += [[start]]
                else:
                    combinator_list += [[x for x in range(start, stop + 1, (stop - start) // steps)]]
            elif(isinstance(entries[0], float)):
                start = float(entries[i * 3])
                stop = float(entries[i * 3 + 1])
                steps = int(entries[i * 3 + 2])
                if(steps == 0 or self.check_whether_excluded(self._modelname, self._key)):
                    combinator_list += [[start]]
                else:
                    combinator_list += [[x for x in float_range(start, stop + 1, (stop - start) // steps)]]
            else:
                print(f"self.key={self._key}, {entries[0]}")
                raise Exception("List element type unsupported")
        list_to_append = [list(x) for x in itertools.product(*combinator_list)]
        return self._chain_object.generate_variants() + [(self._key, list_to_append)]
    def load_csv_list(self):
        _, entries = load_csv(self._modelname, self._key, payload=-1)
        return entries
        
class CombinatorString(Decorator):
    def generate_variants(self):
        setting = self.load_csv_string()
        path = os.path.join(CLASSES_DIRECTORY, self._key + CLASS_EXTENSION)
        if self.check_whether_excluded(self._modelname, self._key):
            return self._chain_object.generate_variants() + [(self._key, [setting])]
        else:
            if(os.path.isdir(path)):
                entries = [x for x in sorted(os.listdir(path))]
                return self._chain_object.generate_variants() + [(self._key, entries)]
            elif(os.path.isfile(path)):
                entries = []
                with open(path, "r") as f:
                    entries = [x.rstrip("\n") for x in f]
                return self._chain_object.generate_variants() + [(self._key, entries)]

            raise Exception(f"Could not find values for string-key: {self._key}")
            
    def load_csv_string(self):
        _, entries = load_csv(self._modelname, self._key, payload=1)
        return entries[0]

class CombinatorPredefined(Decorator):
    def get_variants_with_type(self, conversionFunction):
        if self.check_whether_excluded(self._modelname, self._key):
            return [conversionFunction(x) for x in self.load_csv_predefined()[0:0]]
        else:
            return [conversionFunction(x) for x in self.load_csv_predefined()]
    def load_csv_predefined(self):
        _, entries = load_csv(self._modelname, self._key, payload=-1)
        return entries

class CombinatorPredefinedInteger(CombinatorPredefined):
    def generate_variants(self):
        return self._chain_object.generate_variants() + [(self._key, self.get_variants_with_type(int))]
            

class CombinatorPredefinedFloat(CombinatorPredefined):
    def generate_variants(self):
        return self._chain_object.generate_variants() + [(self._key, self.get_variants_with_type(float))]      
        
def get_key(pair):
    key, _ = pair
    return key
    
def get_value(pair):
    _, value = pair
    return value
        
def filter_elements(variant, to_reduce):
    return_list = []
    for key, entry in variant:
        if(not (key in to_reduce)):
            return_list += [(key, entry)]
    return return_list
            
def reduce_element(variant):
    to_reduce = set()
    for key, entry in variant:
        settings_to_remove = set()
        folder_path = os.path.join(CLASSES_DIRECTORY, key + CLASS_EXTENSION)
        if(isinstance(entry, str) and os.path.isdir(folder_path)):
            for subfile in os.listdir(folder_path):
                file_path = os.path.join(folder_path, subfile)
                with open(file_path, "r") as f:
                    for line in f:
                        settings_to_remove.add(line.rstrip("\n"))
            relevant_file = os.path.join(folder_path, entry)
            with open(relevant_file, "r") as f:
                for line in f:
                    settings_to_remove.remove(line.rstrip("\n"))
        to_reduce = to_reduce.union(settings_to_remove)
        
    return filter_elements(variant, to_reduce)

def remap_keys(variants, keys):
    new_variants = []
    for variant in variants: 
        new_variant = []
        for i, key in enumerate(keys):
            new_pair = (key, variant[i])
            new_variant += [new_pair]
        new_variants += [new_variant]
        
    return new_variants

# From stackoverflow: uniq and sort_and_deduplicate: https://stackoverflow.com/questions/13464152/typeerror-unhashable-type-list-when-using-built-in-set-function
def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item

def sort_and_deduplicate(l):
    return list(uniq(sorted(l, reverse=True)))
            
def variants_from_valuefile(modelname, filepath):
    a = TailClass()
    with open(filepath, "r") as f:
        for line in f:
            key = line.split(VALUE_FILE_DELIMITER)[0]
            a = Decorator.factory(a, modelname, key)
    variants = itertools.product(*(map(get_value, a.generate_variants())))
    keys = [x for x in map(get_key, a.generate_variants())]
    a = list(map(reduce_element, remap_keys(variants, keys)))
    print(f"Count before Optimization={len(a)}")
    a = sort_and_deduplicate(a)
    a = list(a)
    print(f"Count after Optimization={len(a)}")
    return a


    
        
