'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-27

@description: Tabular models for regime classification.
'''

from .xgb import XGBConfig, make_model as make_xgb_model, fit as fit_xgb, predict as predict_xgb, predict_proba as predict_proba_xgb
from .tabnet import TabNetConfig, make_model as make_tabnet_model, fit as fit_tabnet, predict as predict_tabnet, predict_proba as predict_proba_tabnet
from .ft_transformer import (
    FTTransformerConfig,
    make_model as make_fttransformer_model,
    fit as fit_fttransformer,
    predict as predict_fttransformer,
    predict_proba as predict_proba_fttransformer,
)
from .tabpfn import (
    TabPFNConfig,
    make_model as make_tabpfn_model,
    fit as fit_tabpfn,
    predict as predict_tabpfn,
    predict_proba as predict_proba_tabpfn,
)

__all__ = [
    "XGBConfig",
    "make_xgb_model",
    "fit_xgb",
    "predict_xgb",
    "predict_proba_xgb",
    "TabNetConfig",
    "make_tabnet_model",
    "fit_tabnet",
    "predict_tabnet",
    "predict_proba_tabnet",
    "FTTransformerConfig",
    "make_fttransformer_model",
    "fit_fttransformer",
    "predict_fttransformer",
    "predict_proba_fttransformer",
    "TabPFNConfig",
    "make_tabpfn_model",
    "fit_tabpfn",
    "predict_tabpfn",
    "predict_proba_tabpfn",
]
