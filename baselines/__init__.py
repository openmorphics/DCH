"""Optional baselines package.

Import-safe: does not import heavy optional deps (torch, norse, bindsnet) at module import time.
"""
__all__ = ["norse_sg", "bindsnet_stdp"]
