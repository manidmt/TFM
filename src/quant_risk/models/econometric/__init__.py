'''
@author: Manuel Díaz-Meco Terrés

@email: manidmt5@gmail.com

@date: 2026-02-10

@description: Econometric models for quantitative risk analysis.
'''

from .sarimax import SarimaxConfig, fit_sarimax, make_sarimax_features
from .garch import GarchConfig, fit_garch, make_garch_features
from .egarch import EgarchConfig, fit_egarch, make_egarch_features
from .gjrgarch import GjrGarchConfig, fit_gjrgarch, make_gjrgarch_features

__all__ = [
	"SarimaxConfig",
	"fit_sarimax",
	"make_sarimax_features",
	"GarchConfig",
	"fit_garch",
	"make_garch_features",
	"EgarchConfig",
	"fit_egarch",
	"make_egarch_features",
	"GjrGarchConfig",
	"fit_gjrgarch",
	"make_gjrgarch_features",
]
