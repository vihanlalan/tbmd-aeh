"""
TBMD Framework — Source Package
Operationalizing the Adaptive Markets Hypothesis

Author: Vihan Lalan
"""

from .config import SEED, REGIME_PARAMS, WF_CONFIG, BPC_CONFIG, RES_CONFIG
from .simulation import simulate_regime_switching_market
from .bpc import build_behavioral_proxy_composite, build_bpc_real
from .res import compute_real_time_efficiency_score, compute_res
from .validation import walk_forward_validation, walk_forward_validation_real
from .statistics import (
    deflated_sharpe_ratio,
    variance_ratio_test,
    benjamini_hochberg_correction,
    compute_performance_metrics,
    validate_regime_detection,
    validate_directional_claims,
)
