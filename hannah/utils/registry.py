#
# Copyright (c) 2024 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
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
class Registry:
    def __init__(self, name=""):
        self._name = name
        self.registered_classes = {}

    def register(self, cls):
        self.registered_classes[cls.__name__] = cls

    def instantiate(self, name, *args, **kwargs):
        return self.registered_classes[name](*args, **kwargs)

    def __iter__(self):
        return iter(self.registered_classes.values())

    def __len__(self):
        return len(self.registered_classes)

    def __getitem__(self, key):
        return self.registered_classes[key]

    def __contains__(self, key):
        return key in self.registered_classes

    def __repr__(self):
        return f"Registry({self._name}, {self.registered_classes})"

    def __str__(self):
        return f"Registry({self._name}, {self.registered_classes})"

    def keys(self):
        return self.registered_classes.keys()

    def values(self):
        return self.registered_classes.values()

    def items(self):
        return self.registered_classes.items()

    def get(self, key, default=None):
        return self.registered_classes.get(key, default)
