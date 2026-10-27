# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge training"""

# Editable Megatron-Core installs expose megatron.core but not the sibling
# megatron.training package. Include the source submodule for source checkouts.
from pathlib import Path
import sys

_megatron_source = Path(__file__).resolve().parent.parent / "third_party" / "Loong-Megatron"
if (_megatron_source / "megatron" / "training").is_dir() and str(_megatron_source) not in sys.path:
    sys.path.insert(0, str(_megatron_source))
