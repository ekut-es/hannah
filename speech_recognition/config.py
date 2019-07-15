from collections import ChainMap
import argparse
import hashlib
import sys

class ConfigOption(object):
    def __init__(self, nargs=None,
                 const=None, default=None,
                 dtype=None, choices=None,
                 required=False, desc="",
                 category="Global Options", visible=True):

        if default is None and dtype is None:
            raise Exception("Either default or dtype need to be given for a config option.")
        
        self.nargs = nargs
        self.const = const
        self.default = default
        if dtype:
            self.dtype = dtype
        else:
            self.dtype = type(default)
        self.choices = choices
        self.required = required
        self.desc = desc
        self.category = category
        self.visible = visible

    def _get_dict(self, name):
        res = {}
        if self.dtype != bool:
            res["type"] = self.dtype
        if self.default is not None:
            res["default"] = self.default

        if not self.visible: 
            help = argparse.SUPPRESS
        elif self.desc is not None:
            help = str(self.desc)
            if self.default is not None:
                help += " DEFAULT: " + str(self.default)
            res["help"] = help

        if self.choices is not None:
            res["choices"] = self.choices
            
        if self.dtype == bool:
            if self.default == True:
                res["action"] = 'store_false'
                res["dest"] = name
            else:
                res["action"] = 'store_true'
                res["default"] = False

        elif self.dtype == list:
            res["nargs"] = "+"
            res["type"] = type(self.default[0])

        elif self.dtype == tuple:
            res["nargs"] = len(self.default)
            res["type"] = type(self.default[0])
                
        return res

    def get_args(self, name):

        args = self._get_dict(name)
        
        if self.dtype == bool and self.default == True:
            name = "no_" + name
            
        flags = []
        if "_" in name:
            flags.append("--{}".format(name.replace("_", "-")))
        flags.append("--{}".format(name))

        return flags, args
        
class ConfigBuilder(object):
    
    unhashed_options = set(["config", "dataset_cls", "model_class", "type", "cuda", "gpu_no", "output_dir", "config_hash"])
    
    def __init__(self, *default_configs):
        self.default_config = ChainMap(*default_configs)

    def build_argparse(self, parser=None):
        if not parser:
            parser = argparse.ArgumentParser()
        parser.add_argument("--full-help", action="store_true")
        categories = {}
        for key, value in self.default_config.items():
            #Allow overiding of default options
            if not isinstance(value, ConfigOption):
                for map in self.default_config.maps:
                    if key == "cache_size":
                        print("map:")
                        from pprint import pprint
                        pprint(map)
                    if key in map:
                        obj = map[key]
                        if isinstance(obj, ConfigOption):
                            assert type(value) == obj.dtype
                            obj.default = value
                            value = obj
                            break

            flag = "--{}".format(key)
            if isinstance(value, ConfigOption):
                flags, args = value.get_args(key)
                category = parser
                if value.category is not None:
                    if value.category not in categories:
                        category = parser.add_argument_group(title=value.category)
                        categories[value.category] = category
                    category = categories[value.category]
                category.add_argument(*flags, **args) 
            elif isinstance(value, tuple):
                parser.add_argument(flag, default=list(value), nargs=len(value), type=type(value[0]))
            elif isinstance(value, list):
                parser.add_argument(flag, default=value, nargs="+", type=type(value[0]))
            elif isinstance(value, bool):
                if not value:
                    parser.add_argument(flag, action="store_true")
                else:
                    flag = "--no_{}".format(key)
                    parser.add_argument(flag, dest=str(key), action="store_false", default=True)
            else:
                parser.add_argument(flag, default=value, type=type(value))
        return parser

    def config_from_argparse(self, parser=None):
        if not parser:
            parser = self.build_argparse()
        args = parser.parse_args()
        args = vars(args)
        if args["full_help"]:
            parser.print_help()
            sys.exit(0)
            
        config = ChainMap(args, self.default_config)
        config_string = str([item for item in sorted(config.items()) if item[0] not in self.unhashed_options])
            
            
        m = hashlib.sha256()
        m.update(config_string.encode('utf-8'))
        config["config_hash"] = m.hexdigest()
            
        return config
