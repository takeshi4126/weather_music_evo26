# Copyright 2020 The Magenta Authors.
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

# Lint as: python3
"""Replacement for `tf.contrib.training`."""

#from tensor2tensor.utils.hparam import HParams  # pylint:disable=unused-import



class HParams(dict):
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, value):
        self[name] = value
    def values(self):
        return self.items()
    def parse(self, hparams_str):
        if not hparams_str:
            return
        for item in hparams_str.split(','):
            if not item.strip():
                continue
            key, value = item.strip().split('=')
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
            self[key] = value

