'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-10

@description: Econometric models for quantitative risk analysis.
'''

from .sarimax import SarimaxConfig, fit_sarimax, make_sarimax_features

__all__ = ["SarimaxConfig", "fit_sarimax", "make_sarimax_features"]
