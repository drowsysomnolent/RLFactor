import pandas as pd
import numpy as np
from abc import ABC

class RollingFunction(ABC):
    def __init__(self, window=20, min_periods=10, group_id=None):
        self.window = window
        self.min_periods = min_periods
        self.group_id = group_id

    def compute(self, *args):
        pass
    
    def __call__(self, *args):
        return self.compute(*args)


class UnaryRollingFunction(RollingFunction):
    def compute(self, s):
        grouped = s.groupby(level = 1 if self.group_id is None else self.group_id)
        result_list = []
        result = pd.Series(index=s.index, dtype='float64')
        for group_name, group_data in grouped:
            if len(group_data) < self.window:
                group_result = pd.Series(np.zeros(len(group_data)), index=group_data.index)
            else:
                group_result = self._compute(group_data)
            result_list.append(group_result)
        result = pd.concat(result_list).astype(float)
        return result

    def _compute(self, s):
        raise NotImplementedError("Subclass must implement _compute method")


class BinaryRollingFunction(RollingFunction):
    def compute(self, x, y):
        if len(x) != len(y):
            raise ValueError(f"Input series x:{len(x)} and y:{len(y)} have different lengths")
        grouped_x = x.groupby(level = 1 if self.group_id is None else self.group_id)
        grouped_y = y.groupby(level = 1 if self.group_id is None else self.group_id)
        result_list = []
        result = pd.Series(index=x.index, dtype='float64')
        
        for group_name in grouped_x.groups:
            if group_name not in grouped_y.groups:
                raise ValueError(f"Group {group_name} does not exist in y series")
            
            x_group = grouped_x.get_group(group_name)
            y_group = grouped_y.get_group(group_name)
            
            if len(x_group) < self.window or len(y_group) < self.window:
                group_result = pd.Series(np.zeros(len(x_group)), index=x_group.index)
            else:
                group_result = self._compute(x_group, y_group)
            
            result_list.append(group_result)
        
        result = pd.concat(result_list).astype(float)
        return result

    def _compute(self, x, y):
        raise NotImplementedError("Subclass must implement _compute method")


class RollingMin(UnaryRollingFunction):
    def _compute(self, s):
        return s.rolling(self.window, min_periods=self.min_periods).min().fillna(0)

class RollingMax(UnaryRollingFunction):
    def _compute(self, s):
        return s.rolling(self.window, min_periods=self.min_periods).max().fillna(0)

class RollingMean(UnaryRollingFunction):
    def _compute(self, s):
        return s.rolling(self.window, min_periods=self.min_periods).mean().fillna(0)

class RollingDiff(UnaryRollingFunction):
    def _compute(self, s):
        return s.diff().fillna(0)

class RollingStd(UnaryRollingFunction):
    def _compute(self, s):
        return s.rolling(self.window, min_periods=self.min_periods).std().fillna(0)
    

class RollingCorr(BinaryRollingFunction):
    def _compute(self, x, y):
        return x.rolling(self.window, min_periods=self.min_periods).corr(y).fillna(0)

class RollingRegressionBetaK(BinaryRollingFunction):
    def _compute(self, x, y):
        cov = x.rolling(self.window, min_periods=self.min_periods).cov(y).fillna(0)
        var = x.rolling(self.window, min_periods=self.min_periods).var().fillna(0)
        return (cov / var).fillna(0)

class RollingRegressionBetaB(BinaryRollingFunction):
    def _compute(self, x, y):
        mean_x = x.rolling(self.window, min_periods=self.min_periods).mean().fillna(0)
        mean_y = y.rolling(self.window, min_periods=self.min_periods).mean().fillna(0)
        beta_k = cov = x.rolling(self.window, min_periods=self.min_periods).cov(y).fillna(0)
        var = x.rolling(self.window, min_periods=self.min_periods).var().fillna(0)
        beta_k = (cov / var).fillna(0)
        return (mean_y - beta_k * mean_x).fillna(0)


def add(x, y):
    if isinstance(x, (int, float)) and isinstance(y, pd.Series):
        return y + x
    elif isinstance(y, (int, float)) and isinstance(x, pd.Series):
        return x + y
    else:
        return x + y

def sub(x, y):
    if isinstance(x, (int, float)) and isinstance(y, pd.Series):
        return -y + x
    elif isinstance(y, (int, float)) and isinstance(x, pd.Series):
        return x - y
    else:
        return x - y

def mul(x, y):
    if isinstance(x, (int, float)) and isinstance(y, pd.Series):
        return y * x
    elif isinstance(y, (int, float)) and isinstance(x, pd.Series):
        return x * y
    else:
        return x * y

def div(x, y):
    if isinstance(x, (int, float)) and isinstance(y, pd.Series):
        return x / y
    elif isinstance(y, (int, float)) and isinstance(x, pd.Series):
        return x / y
    else:
        return x / y
    
operater_dict = {'add': add, 'sub': sub, 'mul': mul, 'div': div}

unary_rolling = {
    'rolling_mean': RollingMean,
    'rolling_std': RollingStd,
    'rolling_max': RollingMax,
    'rolling_min': RollingMin,
    'rolling_diff': RollingDiff
}

binary_rolling = {
    'rolling_corr': RollingCorr,
    'rolling_regression_beta_B': RollingRegressionBetaB,
    'rolling_regression_beta_k': RollingRegressionBetaK
}
