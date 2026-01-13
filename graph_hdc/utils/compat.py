"""
Compatibility utilities for loading checkpoints from original research repo.

The original research repo used 'src' as the package name. This module provides
utilities to remap module paths when loading those checkpoints.
"""

import sys
import types


def setup_module_aliases() -> None:
    """
    Set up module aliases for loading checkpoints from original research repo.

    The original repo used 'src' as the package name with structure:
    - src.models.flows.real_nvp -> graph_hdc.models.flows.real_nvp
    - src.exp.logp_regressor.pr -> graph_hdc.models.regressors.property_regressor
    - src.exp.real_nvp_hpo.* -> graph_hdc.models.flows.real_nvp
    - src.encoding.* -> graph_hdc.hypernet.*
    - src.normalizing_flow.* -> graph_hdc.models.flows.*

    This function creates mock modules in sys.modules to allow pickle to find
    the classes when deserializing checkpoints.
    """
    if "src" in sys.modules:
        # Already set up
        return

    import graph_hdc.models.flows.real_nvp as flows_module
    import graph_hdc.models.regressors.property_regressor as regressor_module
    import graph_hdc.hypernet.encoder as encoder_module
    import graph_hdc.hypernet.configs as configs_module
    import graph_hdc.hypernet.types as types_module

    # Create the full src module hierarchy
    src = types.ModuleType("src")

    # src.models
    src.models = types.ModuleType("src.models")
    src.models.flows = types.ModuleType("src.models.flows")
    src.models.flows.real_nvp = flows_module

    # src.exp
    src.exp = types.ModuleType("src.exp")
    src.exp.logp_regressor = types.ModuleType("src.exp.logp_regressor")
    src.exp.logp_regressor.pr = regressor_module
    src.exp.real_nvp_hpo = types.ModuleType("src.exp.real_nvp_hpo")
    src.exp.real_nvp_hpo.real_nvp_composite = flows_module
    src.exp.real_nvp_hpo.real_nvp_v3_composite = flows_module

    # src.encoding -> graph_hdc.hypernet
    src.encoding = types.ModuleType("src.encoding")
    src.encoding.graph_encoders = encoder_module
    src.encoding.configs_and_constants = configs_module
    src.encoding.the_types = types_module

    # src.normalizing_flow -> graph_hdc.models.flows
    src.normalizing_flow = types.ModuleType("src.normalizing_flow")
    src.normalizing_flow.real_nvp = flows_module

    # Register all in sys.modules
    sys.modules["src"] = src
    sys.modules["src.models"] = src.models
    sys.modules["src.models.flows"] = src.models.flows
    sys.modules["src.models.flows.real_nvp"] = flows_module
    sys.modules["src.exp"] = src.exp
    sys.modules["src.exp.logp_regressor"] = src.exp.logp_regressor
    sys.modules["src.exp.logp_regressor.pr"] = regressor_module
    sys.modules["src.exp.real_nvp_hpo"] = src.exp.real_nvp_hpo
    sys.modules["src.exp.real_nvp_hpo.real_nvp_composite"] = flows_module
    sys.modules["src.exp.real_nvp_hpo.real_nvp_v3_composite"] = flows_module
    sys.modules["src.encoding"] = src.encoding
    sys.modules["src.encoding.graph_encoders"] = encoder_module
    sys.modules["src.encoding.configs_and_constants"] = configs_module
    sys.modules["src.encoding.the_types"] = types_module
    sys.modules["src.normalizing_flow"] = src.normalizing_flow
    sys.modules["src.normalizing_flow.real_nvp"] = flows_module
