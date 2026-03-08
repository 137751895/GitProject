# ===== 修改说明 =====
# 依据《当前项目性能提升指南-最终版v2.md》改进点 1.3
# 新增内容已用 # [新增] 标记
# ===================
"""
商品期货机器学习量化模型 - VaR与压力测试模块
Value at Risk and Stress Testing for tail risk management.
"""

from __future__ import annotations  # [新增]

import logging  # [新增]
from dataclasses import dataclass  # [新增]
from typing import Dict, List, Optional  # [新增]

import numpy as np  # [新增]
from scipy import stats  # [新增]


@dataclass  # [新增]
class VaRResult:  # [新增]
    """Value at Risk calculation result"""  # [新增]
    confidence_level: float  # [新增]
    time_horizon: str  # [新增]
    var_value: float  # [新增]
    expected_shortfall: float  # [新增]
    method: str  # [新增]
    portfolio_value: float  # [新增]
    historical_var: Optional[float] = None  # [新增]
    parametric_var: Optional[float] = None  # [新增]
    monte_carlo_var: Optional[float] = None  # [新增]


@dataclass  # [新增]
class StressTestResult:  # [新增]
    """Stress test result"""  # [新增]
    scenario_name: str  # [新增]
    portfolio_value_before: float  # [新增]
    portfolio_value_after: float  # [新增]
    percentage_loss: float  # [新增]
    risk_metrics: Dict[str, float]  # [新增]
    recovery_period: Optional[int] = None  # [新增]
    tail_risk: Optional[float] = None  # [新增]


class RiskModels:  # [新增]
    """Comprehensive risk analysis models"""  # [新增]

    def __init__(self):  # [新增]
        self.logger = logging.getLogger(__name__)  # [新增]

    def calculate_historical_var(self, returns: List[float], confidence_level: float = 0.95,
                                portfolio_value: float = 1000000) -> Optional[VaRResult]:  # [新增]
        if len(returns) < 30:  # [新增]
            self.logger.warning("Insufficient data for Historical VaR calculation")  # [新增]
            return None  # [新增]

        try:  # [新增]
            returns_array = np.array(returns)  # [新增]
            var_percentile = (1 - confidence_level) * 100  # [新增]
            var_value = np.percentile(returns_array, var_percentile) * portfolio_value  # [新增]

            tail_returns = returns_array[returns_array <= np.percentile(returns_array, var_percentile)]  # [新增]
            expected_shortfall = np.mean(tail_returns) * portfolio_value if len(tail_returns) > 0 else var_value  # [新增]

            return VaRResult(  # [新增]
                confidence_level=confidence_level,  # [新增]
                time_horizon="1d",  # [新增]
                var_value=abs(float(var_value)),  # [新增]
                expected_shortfall=abs(float(expected_shortfall)),  # [新增]
                method="historical",  # [新增]
                portfolio_value=float(portfolio_value),  # [新增]
                historical_var=abs(float(var_value)),  # [新增]
            )  # [新增]

        except Exception as e:  # [新增]
            self.logger.error(f"Error calculating Historical VaR: {str(e)}")  # [新增]
            return None  # [新增]

    def calculate_parametric_var(self, returns: List[float], confidence_level: float = 0.95,
                                portfolio_value: float = 1000000) -> Optional[VaRResult]:  # [新增]
        if len(returns) < 30:  # [新增]
            self.logger.warning("Insufficient data for Parametric VaR calculation")  # [新增]
            return None  # [新增]

        try:  # [新增]
            returns_array = np.array(returns)  # [新增]
            mean_return = np.mean(returns_array)  # [新增]
            std_return = np.std(returns_array, ddof=1)  # [新增]  # [BUGFIX] P2-3: sample std

            z_score = stats.norm.ppf(1 - confidence_level)  # [新增]
            var_value = (mean_return + z_score * std_return) * portfolio_value  # [新增]
            expected_shortfall = (mean_return - std_return * stats.norm.pdf(z_score) / (1 - confidence_level)) * portfolio_value  # [新增]

            return VaRResult(  # [新增]
                confidence_level=confidence_level,  # [新增]
                time_horizon="1d",  # [新增]
                var_value=abs(float(var_value)),  # [新增]
                expected_shortfall=abs(float(expected_shortfall)),  # [新增]
                method="parametric",  # [新增]
                portfolio_value=float(portfolio_value),  # [新增]
                parametric_var=abs(float(var_value)),  # [新增]
            )  # [新增]

        except Exception as e:  # [新增]
            self.logger.error(f"Error calculating Parametric VaR: {str(e)}")  # [新增]
            return None  # [新增]

    def calculate_monte_carlo_var(self, returns: List[float], confidence_level: float = 0.95,
                                 portfolio_value: float = 1000000, simulations: int = 10000,
                                 random_state: int = 42) -> Optional[VaRResult]:  # [新增]  # [BUGFIX] P0-3: accept seed
        if len(returns) < 30:  # [新增]
            self.logger.warning("Insufficient data for Monte Carlo VaR calculation")  # [新增]
            return None  # [新增]

        try:  # [新增]
            returns_array = np.array(returns)  # [新增]
            mean_return = np.mean(returns_array)  # [新增]
            std_return = np.std(returns_array, ddof=1)  # [新增]  # [BUGFIX] P2-3: sample std

            rng = np.random.RandomState(random_state)  # [新增]  # [BUGFIX] P0-3: reproducible
            simulated_returns = rng.normal(mean_return, std_return, simulations)  # [新增]  # [BUGFIX] P0-3
            var_percentile = (1 - confidence_level) * 100  # [新增]
            var_value = np.percentile(simulated_returns, var_percentile) * portfolio_value  # [新增]

            tail_returns = simulated_returns[simulated_returns <= np.percentile(simulated_returns, var_percentile)]  # [新增]
            expected_shortfall = np.mean(tail_returns) * portfolio_value if len(tail_returns) > 0 else var_value  # [新增]

            return VaRResult(  # [新增]
                confidence_level=confidence_level,  # [新增]
                time_horizon="1d",  # [新增]
                var_value=abs(float(var_value)),  # [新增]
                expected_shortfall=abs(float(expected_shortfall)),  # [新增]
                method="monte_carlo",  # [新增]
                portfolio_value=float(portfolio_value),  # [新增]
                monte_carlo_var=abs(float(var_value)),  # [新增]
            )  # [新增]

        except Exception as e:  # [新增]
            self.logger.error(f"Error calculating Monte Carlo VaR: {str(e)}")  # [新增]
            return None  # [新增]

    def calculate_comprehensive_var(self, returns: List[float], confidence_levels: Optional[List[float]] = None,
                                    portfolio_value: float = 1000000) -> List[VaRResult]:  # [新增]
        if confidence_levels is None:  # [新增]
            confidence_levels = [0.95, 0.99]  # [新增]

        results = []  # [新增]

        for confidence_level in confidence_levels:  # [新增]
            historical_var = self.calculate_historical_var(returns, confidence_level, portfolio_value)  # [新增]
            parametric_var = self.calculate_parametric_var(returns, confidence_level, portfolio_value)  # [新增]
            monte_carlo_var = self.calculate_monte_carlo_var(returns, confidence_level, portfolio_value)  # [新增]

            if historical_var and parametric_var and monte_carlo_var:  # [新增]
                comprehensive_var = VaRResult(  # [新增]
                    confidence_level=float(confidence_level),  # [新增]
                    time_horizon="1d",  # [新增]
                    var_value=float(np.mean([historical_var.var_value, parametric_var.var_value, monte_carlo_var.var_value])),  # [新增]
                    expected_shortfall=float(np.mean([historical_var.expected_shortfall, parametric_var.expected_shortfall, monte_carlo_var.expected_shortfall])),  # [新增]
                    method="comprehensive",  # [新增]
                    portfolio_value=float(portfolio_value),  # [新增]
                    historical_var=float(historical_var.var_value),  # [新增]
                    parametric_var=float(parametric_var.var_value),  # [新增]
                    monte_carlo_var=float(monte_carlo_var.var_value),  # [新增]
                )  # [新增]
                results.append(comprehensive_var)  # [新增]

        return results  # [新增]

    def perform_stress_testing(self, current_returns: List[float], scenarios: Dict[str, Dict[str, float]],
                              portfolio_value: float = 1000000) -> List[StressTestResult]:  # [新增]
        results = []  # [新增]

        try:  # [新增]
            base_metrics = self._calculate_risk_metrics(current_returns)  # [新增]

            for scenario_name, scenario_params in scenarios.items():  # [新增]
                stressed_returns = self._apply_stress_scenario(current_returns, scenario_params)  # [新增]

                cumulative_return = np.prod(1 + np.array(stressed_returns)) - 1  # [新增]
                stressed_value = portfolio_value * (1 + cumulative_return)  # [新增]

                stressed_metrics = self._calculate_risk_metrics(stressed_returns)  # [新增]
                tail_risk = self._calculate_tail_risk(stressed_returns)  # [新增]

                result = StressTestResult(  # [新增]
                    scenario_name=scenario_name,  # [新增]
                    portfolio_value_before=float(portfolio_value),  # [新增]
                    portfolio_value_after=float(stressed_value),  # [新增]
                    percentage_loss=float(cumulative_return * 100),  # [新增]
                    risk_metrics=stressed_metrics,  # [新增]
                    recovery_period=self._estimate_recovery_period(stressed_returns),  # [新增]
                    tail_risk=float(tail_risk),  # [新增]
                )  # [新增]

                results.append(result)  # [新增]

        except Exception as e:  # [新增]
            self.logger.error(f"Error performing stress testing: {str(e)}")  # [新增]

        return results  # [新增]

    def _apply_stress_scenario(self, returns: List[float], scenario_params: Dict[str, float]) -> List[float]:  # [新增]
        stressed_returns = []  # [新增]

        for ret in returns:  # [新增]
            stressed_return = ret  # [新增]

            if 'shock' in scenario_params:  # [新增]
                stressed_return += scenario_params['shock']  # [新增]

            if 'volatility_multiplier' in scenario_params:  # [新增]
                stressed_return *= scenario_params['volatility_multiplier']  # [新增]

            if 'correlation_breakdown' in scenario_params and scenario_params['correlation_breakdown']:  # [新增]
                rng = np.random.RandomState(42)  # [BUGFIX] P1-3: deterministic stress test
                stressed_return += rng.normal(0, abs(stressed_return) * 0.5)  # [新增]  # [BUGFIX] P1-3

            stressed_returns.append(stressed_return)  # [新增]

        return stressed_returns  # [新增]

    def _calculate_risk_metrics(self, returns: List[float]) -> Dict[str, float]:  # [新增]
        if not returns:  # [新增]
            return {}  # [新增]

        try:  # [新增]
            returns_array = np.array(returns)  # [新增]
            std_val = float(np.std(returns_array))  # [新增]
            var_95_threshold = float(np.percentile(returns_array, 5))  # [新增]  # [BUGFIX] P0-1
            tail = returns_array[returns_array <= var_95_threshold]  # [新增]  # [BUGFIX] P0-1
            es_95 = float(np.mean(tail)) if len(tail) > 0 else var_95_threshold  # [新增]  # [BUGFIX] P0-1
            metrics = {  # [新增]
                'volatility': std_val,  # [新增]
                'sharpe_ratio': float(np.mean(returns_array) / std_val) if std_val > 0 else 0.0,  # [新增]
                'max_drawdown': float(self._calculate_max_drawdown(returns_array)),  # [新增]
                'skewness': float(stats.skew(returns_array)),  # [新增]
                'kurtosis': float(stats.kurtosis(returns_array)),  # [新增]
                'value_at_risk_95': var_95_threshold,  # [新增]  # [BUGFIX] P0-1
                'expected_shortfall_95': es_95,  # [新增]  # [BUGFIX] P0-1
            }  # [新增]

            return metrics  # [新增]

        except Exception as e:  # [新增]
            self.logger.error(f"Error calculating risk metrics: {str(e)}")  # [新增]
            return {}  # [新增]

    def _calculate_tail_risk(self, returns: List[float]) -> float:  # [新增]
        returns_array = np.array(returns)  # [新增]
        if len(returns_array) < 2:  # [BUGFIX] P0-2: guard empty/tiny arrays
            return 0.0  # [BUGFIX] P0-2
        var_95 = np.percentile(returns_array, 5)  # [新增]
        tail_returns = returns_array[returns_array <= var_95]  # [新增]
        if len(tail_returns) > 1:  # [新增]  # [BUGFIX] P0-2: need >1 for std
            return float(np.std(tail_returns))  # [新增]
        return 0.0  # [新增]

    def _estimate_recovery_period(self, stressed_returns: List[float]) -> Optional[int]:  # [新增]
        cumulative_returns = np.cumprod(1 + np.array(stressed_returns))  # [新增]
        peak = np.maximum.accumulate(cumulative_returns)  # [新增]
        drawdown = (peak - cumulative_returns) / peak  # [新增]
        recovery_points = np.where(drawdown < 0.05)[0]  # [新增]
        if len(recovery_points) > 0:  # [新增]
            return int(recovery_points[0])  # [新增]
        return None  # [新增]

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:  # [新增]
        cumulative = np.cumprod(1 + returns)  # [新增]
        peak = np.maximum.accumulate(cumulative)  # [新增]
        drawdown = (peak - cumulative) / peak  # [新增]
        return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0  # [新增]
