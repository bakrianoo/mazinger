# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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
from .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

# [mazinger] The 25 Hz tokenizer imports `sox` and `onnxruntime` at module
# scope, and pulls in the `sox` *binary* as a runtime requirement.  Mazinger
# only ever loads the 12 Hz models, so resolving these names is deferred
# (PEP 562) instead of paying for them on every import.  Behaviour is
# unchanged for callers that do touch the 25 Hz path.
_LAZY_25HZ = {
    "Qwen3TTSTokenizerV1Config": ".tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1",
    "Qwen3TTSTokenizerV1Model": ".tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1",
}


def __getattr__(name):
    module = _LAZY_25HZ.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted([*globals(), *_LAZY_25HZ])