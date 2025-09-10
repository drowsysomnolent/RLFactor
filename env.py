import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import List, Tuple, Dict, Any, Callable, Optional
import copy
from utils import convert_postfix_to_infix

class FactorEnvironment:
    def __init__(
        self,
        features: List[str],
        unary_ops: Dict[str, Any],
        binary_ops: Dict[str, Any],
        operator_dict: Dict[str, Callable],
        df: pd.DataFrame,
        target: str,
        num_factors: int = 10,
        max_expr_length: int = 15,
        min_expr_length: int = 2,
        window_sizes: Optional[List[int]] = None,
        group_id: Optional[pd.Series] = None,
        duplicate_corr_threshold: float = 0.8,
        ic_min_abs: float = 0.02,
        verbose: bool = True
    ):
        self.features = features
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.operator_dict = operator_dict
        self.df = df
        self.target = target
        self.num_factors = num_factors
        self.max_expr_length = max_expr_length
        self.min_expr_length = min_expr_length
        self.window_sizes = window_sizes if window_sizes is not None else [5, 7, 10, 15, 20]
        self.group_id = group_id
        self.verbose = verbose

        self.duplicate_corr_threshold = duplicate_corr_threshold
        self.ic_min_abs = ic_min_abs

        self.factor_pool: List[pd.Series] = []
        self.factor_pool_expressions: List[list] = []
        self.factor_pool_ics: List[float] = []

        self.feature_names = list(self.features)
        self.unary_names = list(self.unary_ops.keys())
        self.binary_names = list(self.binary_ops.keys())
        self.basic_names = list(self.operator_dict.keys())
        
        self.feature_index = {n: i for i, n in enumerate(self.feature_names)}
        self.unary_index = {n: i for i, n in enumerate(self.unary_names)}
        self.binary_index = {n: i for i, n in enumerate(self.binary_names)}
        self.basic_index = {n: i for i, n in enumerate(self.basic_names)}

        self.symbol_map = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '**'}

        self.action_space = self._build_action_space()
        self.action_dim = len(self.action_space)
        self.state_dim = len(self.feature_names) + len(self.unary_names) + len(self.binary_names) + len(self.basic_names) + 2

        self.reset()

    def _build_action_space(self):
        actions = []
        for feat in self.feature_names:
            actions.append(("feature", feat))
        for op_name in self.unary_names:
            for window in self.window_sizes:
                actions.append(("unary", op_name, window))
        for op_name in self.binary_names:
            for window in self.window_sizes:
                actions.append(("binary", op_name, window))
        for op_name in self.basic_names:
            actions.append(("basic_binary", op_name))
        return actions

    def reset(self):
        self.current_factor_idx = 0
        self.factors = self.factor_pool.copy()
        self.factor_expressions = self.factor_pool_expressions.copy()
        
        self.tokens: List[tuple] = []
        self.operand_stack: List[str] = []
        self.done = False
        self.steps = 0
        self.episode_done = False
        
        return self._get_state()

    def _get_state(self):
        feature_usage = [0] * len(self.feature_names)
        unary_usage = [0] * len(self.unary_names)
        binary_usage = [0] * len(self.binary_names)
        basic_usage = [0] * len(self.basic_names)
        
        for token in self.tokens:
            if len(token) >= 5:
                kind = token[3]
                name = token[4]
                if kind == 'feature' and name in self.feature_index:
                    feature_usage[self.feature_index[name]] += 1
                elif kind == 'unary' and name in self.unary_index:
                    unary_usage[self.unary_index[name]] += 1
                elif kind == 'binary' and name in self.binary_index:
                    binary_usage[self.binary_index[name]] += 1
                elif kind == 'basic_binary' and name in self.basic_index:
                    basic_usage[self.basic_index[name]] += 1

        def norm(arr):
            m = max(arr) if arr else 1
            return [a / m if m > 0 else 0 for a in arr]

        feature_usage = norm(feature_usage)
        unary_usage = norm(unary_usage)
        binary_usage = norm(binary_usage)
        basic_usage = norm(basic_usage)

        stack_size = [len(self.operand_stack) / 10.0]
        factor_progress = [self.current_factor_idx / max(1, self.num_factors)]
        
        state = feature_usage + unary_usage + binary_usage + basic_usage + stack_size + factor_progress
        return np.array(state, dtype=np.float32)

    def _is_valid_action(self, action):
        action_type = action[0]
        
        if action_type == "feature":
            return True
        elif action_type == "unary":
            return len(self.operand_stack) >= 1
        elif action_type in ("binary", "basic_binary"):
            return len(self.operand_stack) >= 2
        
        return False

    def get_valid_actions(self):
        return [i for i, action in enumerate(self.action_space) if self._is_valid_action(action)]
    
    def calculate_postfix_expression(self, tokens, df):
        stack = []
        
        for token in tokens:
            func_or_feat, arity = token[0], token[1]
            
            if isinstance(func_or_feat, str):
                stack.append(df[func_or_feat].copy())
            else:
                if arity == 1:
                    if len(stack) < 1:
                        raise ValueError("Not enough operands for unary operation")
                    operand = stack.pop()
                    result = func_or_feat(operand)
                    stack.append(result)
                elif arity == 2:
                    if len(stack) < 2:
                        raise ValueError("Not enough operands for binary operation")
                    b = stack.pop()
                    a = stack.pop()
                    result = func_or_feat(a, b)
                    stack.append(result)
                else:
                    raise ValueError(f"Unsupported arity: {arity}")
        
        if len(stack) != 1:
            raise ValueError("Invalid expression: final stack should contain exactly one element")
        
        return stack[0]

    def calculate_ic(self, factor: pd.Series, target: pd.Series) -> float:
        valid_idx = ~(factor.isna() | target.isna())
        if valid_idx.sum() < 10:
            return 0.0
        f = factor[valid_idx]
        y = target[valid_idx]
        try:
            ic, _ = stats.pearsonr(f, y)
            return ic if not np.isnan(ic) else 0.0
        except:
            return 0.0

    def calculate_correlation(self, s1: pd.Series, s2: pd.Series) -> float:
        valid_idx = ~(s1.isna() | s2.isna())
        if valid_idx.sum() < 10:
            return 0.0
        a, b = s1[valid_idx], s2[valid_idx]
        try:
            corr, _ = stats.pearsonr(a, b)
            return 0.0 if np.isnan(corr) else float(corr)
        except:
            return 0.0

    def is_duplicate_factor(self, factor: pd.Series) -> bool:
        for existing in self.factor_pool:
            corr = self.calculate_correlation(factor, existing)
            if abs(corr) > self.duplicate_corr_threshold:
                return True
        return False

    def _add_to_factor_pool(self, factor: pd.Series, expression: list, ic: float):
        if len(self.factor_pool) < self.num_factors:
            self.factor_pool.append(factor)
            self.factor_pool_expressions.append(expression)
            self.factor_pool_ics.append(ic)
            if self.verbose:
                try:
                    infix = convert_postfix_to_infix(expression)
                except:
                    infix = f"Expression_{len(self.factor_pool)}"
                print(f"✅ Added factor {len(self.factor_pool)}: {infix} (IC={ic:.6f})")
        else:
            worst_idx = np.argmin(np.abs(self.factor_pool_ics))
            worst_ic = self.factor_pool_ics[worst_idx]
            
            if abs(ic) > abs(worst_ic):
                if self.verbose:
                    try:
                        old_expr = convert_postfix_to_infix(self.factor_pool_expressions[worst_idx])
                        new_expr = convert_postfix_to_infix(expression)
                    except:
                        old_expr = f"Factor_{worst_idx}"
                        new_expr = f"New_Factor"
                    print(f"🔄 Replaced factor {worst_idx}: {old_expr} (IC={worst_ic:.6f}) -> {new_expr} (IC={ic:.6f})")
                
                self.factor_pool[worst_idx] = factor
                self.factor_pool_expressions[worst_idx] = expression
                self.factor_pool_ics[worst_idx] = ic

    def step(self, action_idx):
        self.steps += 1
        action = self.action_space[action_idx]

        if not self._is_valid_action(action):
            return self._get_state(), -0.1, self.episode_done, {"valid": False, "factor_done": False}

        t = action[0]
        if t == "feature":
            feat = action[1]
            self.tokens.append((feat, 0, feat, 'feature', feat))
            self.operand_stack.append(feat)

        elif t == "unary":
            op_name, window = action[1], action[2]
            op_class = self.unary_ops[op_name]
            op_func = op_class(window=window, min_periods=max(2, window // 2), group_id=self.group_id)
            token_str = f"{op_name}({window})"
            self.tokens.append((op_func, 1, token_str, 'unary', op_name))
            self.operand_stack.pop()
            self.operand_stack.append("result")

        elif t == "binary":
            op_name, window = action[1], action[2]
            op_class = self.binary_ops[op_name]
            op_func = op_class(window=window, min_periods=max(2, window // 2), group_id=self.group_id)
            token_str = f"{op_name}({window})"
            self.tokens.append((op_func, 2, token_str, 'binary', op_name))
            self.operand_stack = self.operand_stack[:-2] + ["result"]

        elif t == "basic_binary":
            op_name = action[1]
            op_func = self.operator_dict[op_name]
            token_str = self.symbol_map.get(op_name, op_name)
            self.tokens.append((op_func, 2, token_str, 'basic_binary', op_name))
            self.operand_stack = self.operand_stack[:-2] + ["result"]

        if len(self.tokens) >= self.max_expr_length:
            return self._get_state(), -0.1, self.episode_done, {"valid": True, "factor_done": False}

        factor_done = (len(self.operand_stack) == 1) and (len(self.tokens) >= self.min_expr_length)
        reward = 0.0
        if factor_done:
            try:
                result = self.calculate_postfix_expression(self.tokens, self.df)
                result = result.replace([np.inf, -np.inf], np.nan)
                
                std = result.std()
                if pd.isna(std) or std == 0:
                    raise ValueError("Zero or NaN std in factor")
                result = (result - result.mean()) / (std + 1e-8)

                ic = self.calculate_ic(result, self.df[self.target])
                
                if abs(ic) < self.ic_min_abs:
                    reward = -0.05
                elif self.is_duplicate_factor(result):
                    reward = -0.2
                else:
                    reward = abs(ic) * 10.0
                    self._add_to_factor_pool(result, copy.deepcopy(self.tokens), ic)

            except Exception as e:
                if self.verbose:
                    print(f"Factor evaluation error: {e}")
                reward = -0.1

            self.current_factor_idx += 1
            self.tokens = []
            self.operand_stack = []
            self.steps = 0

            if self.current_factor_idx >= self.num_factors:
                self.episode_done = True
                reward += 1.0

            return self._get_state(), reward, self.episode_done, {"valid": True, "factor_done": True}

        return self._get_state(), reward, self.episode_done, {"valid": True, "factor_done": False}

    def get_current_factors(self):
        return self.factor_pool.copy()

    def get_factor_expressions(self):
        return self.factor_pool_expressions.copy()

    def get_factor_ics(self):
        return self.factor_pool_ics.copy()

    def generate_summary(self) -> Dict[str, Any]:
        summary = {
            'num_factors': len(self.factor_pool),
            'target_factors': self.num_factors,
            'factors': []
        }

        print(f"📊 Factor summary:")
        print(f"   Selected factors: {len(self.factor_pool)}/{self.num_factors}")
        
        if self.factor_pool:
            print(f"\n📋 Factor list:")
            for i, (ic, expression) in enumerate(zip(self.factor_pool_ics, self.factor_pool_expressions)):
                try:
                    infix_expr = convert_postfix_to_infix(expression)
                except:
                    infix_expr = f"Factor_{i}"
                
                print(f"   Factor_{i}: IC={ic:7.5f} | {infix_expr}")
                summary['factors'].append({
                    'factor_id': i,
                    'expression': infix_expr,
                    'ic': float(ic)
                })

            ics = [abs(ic) for ic in self.factor_pool_ics]
            print(f"\n📈 Factor quality statistics:")
            print(f"   Avg |IC|: {np.mean(ics):7.5f}")
            print(f"   Max |IC|: {np.max(ics):7.5f}")
            print(f"   Min |IC|: {np.min(ics):7.5f}")
            
            summary['avg_abs_ic'] = float(np.mean(ics))
            summary['max_abs_ic'] = float(np.max(ics))
            summary['min_abs_ic'] = float(np.min(ics))

        return summary