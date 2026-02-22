# Copyright 2023 The Waymo Open Dataset Authors.
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
# ==============================================================================
"""Vendored Waymo v2 dataframe merge utils with optional dask dependency."""

from typing import Set

import pandas as pd

try:
    import dask.dataframe as dd
except Exception:  # pragma: no cover - optional dependency
    dd = None


def _select_key_columns(df, prefix):
    return set([c for c in df.columns if c.startswith(prefix)])


def _how(left_nullable=False, right_nullable=False):
    if left_nullable and right_nullable:
        return "outer"
    if left_nullable and not right_nullable:
        return "right"
    if not left_nullable and right_nullable:
        return "left"
    return "inner"


def _cast_keys(src, dst, keys):
    for key in keys:
        if dst.dtypes[key] != src.dtypes[key]:
            dst[key] = dst[key].astype(src[key].dtype)


def _group_by(src, keys):
    dst = src.groupby(list(keys)).agg(list).reset_index()
    # Fix key types automatically created from the MultiIndex.
    _cast_keys(src, dst, keys)
    return dst


def merge(
    left,
    right,
    left_nullable=False,
    right_nullable=False,
    left_group=False,
    right_group=False,
    key_prefix="key.",
):
    """Merges two tables by auto-selecting key columns."""
    left_keys = _select_key_columns(left, key_prefix)
    right_keys = _select_key_columns(right, key_prefix)
    common_keys = left_keys.intersection(right_keys)
    if left_group and left_keys != common_keys:
        left = _group_by(left, common_keys)
    if right_group and right_keys != common_keys:
        right = _group_by(right, common_keys)
    return left.merge(right, on=list(common_keys), how=_how(left_nullable, right_nullable))

