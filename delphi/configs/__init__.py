# Author: pstefanou12@
"""Pydantic configuration models for delphi.ai algorithms.

The framework-level configs are flat; the model configs are nested by domain,
with truncation as a modifier within each domain. Import a config from the
submodule that defines it, e.g.
``from delphi.configs.stats.truncated import regression``.
"""
