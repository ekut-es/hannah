#
# Copyright (c) 2022 University of Tübingen.
#
# This file is part of hannah.
# See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import importlib
import operator
import types
from importlib.util import find_spec



def _module_available(module_path: str) -> bool:
    """Check if a path is available in your environment.
    >>> _module_available('os')
    True
    >>> _module_available('bla.bla')
    False
    """
    try:
        return find_spec(module_path) is not None
    except AttributeError:
        # Python 3.6
        return False
    except ModuleNotFoundError:
        # Python 3.7+
        return False
    except ValueError:
        # Sometimes __spec__ can be None and gives a ValueError
        return True

_TORCH_AVAILABLE = _module_available("torch")
_KORNIA_AVAILABLE = _module_available("kornia")
_COCO_AVAILABLE = _module_available("pycocotools")
_TIMM_AVAILABLE = _module_available("timm")
_TORCHVISION_AVAILABLE = _module_available("torchvision")
_PYTORCHVIDEO_AVAILABLE = _module_available("pytorchvideo")
_TORCHAUDIO_AVAILABLE = _module_available("torchaudio")
_DATASETS_AVAILABLE = _module_available("datasets")
_LEARN2LEARN_AVAILABLE = _module_available("learn2learn") and _compare_version(
    "learn2learn", operator.ge, "0.1.6"
)
_TORCH_ORT_AVAILABLE = _module_available("torch_ort")
_VISSL_AVAILABLE = _module_available("vissl") and _module_available("classy_vision")
_ALBUMENTATIONS_AVAILABLE = _module_available("albumentations")
_BAAL_AVAILABLE = _module_available("baal")
_TORCH_OPTIMIZER_AVAILABLE = _module_available("torch_optimizer")
_SENTENCE_TRANSFORMERS_AVAILABLE = _module_available("sentence_transformers")


def lazy_import(module_name, callback=None):
    """Returns a proxy module object that will lazily import the given module the first time it is used.
    Example usage::
        # Lazy version of `import tensorflow as tf`
        tf = lazy_import("tensorflow")
        # Other commands
        # Now the module is loaded
        tf.__version__
    Args:
        module_name: the fully-qualified module name to import
        callback (None): a callback function to call before importing the
            module
    Returns:
        a proxy module object that will be lazily imported when first used
    """
    return LazyModule(module_name, callback=callback)


class LazyModule(types.ModuleType):
    """Proxy module that lazily imports the underlying module the first time it is actually used.
    Args:
        module_name: the fully-qualified module name to import
        callback (None): a callback function to call before importing the
            module
    """

    def __init__(self, module_name, callback=None):
        super().__init__(module_name)
        self._module = None
        self._callback = callback

    def __getattr__(self, item):
        if self._module is None:
            self._import_module()

        return getattr(self._module, item)

    def __dir__(self):
        if self._module is None:
            self._import_module()

        return dir(self._module)

    def _import_module(self):
        # Execute callback, if any
        if self._callback is not None:
            self._callback()

        # Actually import the module
        module = importlib.import_module(self.__name__)
        self._module = module

        # Update this object's dict so that attribute references are efficient
        # (__getattr__ is only called on lookups that fail)
        self.__dict__.update(module.__dict__)
