import pandas as pd
import numpy as np
from typing import Dict, Any

def convert_postfix_to_infix(tokens):
    if not tokens:
        return "Empty"
    
    stack = []
    for token in tokens:
        if len(token) >= 3:
            if isinstance(token[0], str):
                stack.append(token[0])
            else:
                op_str = token[2] if len(token) > 2 else "op"
                arity = token[1]
                if arity == 1:
                    if stack:
                        operand = stack.pop()
                        stack.append(f"{op_str}({operand})")
                    else:
                        stack.append(f"{op_str}(?)")
                elif arity == 2:
                    if len(stack) >= 2:
                        b = stack.pop()
                        a = stack.pop()
                        if op_str in ['+', '-', '*', '/', '**']:
                            stack.append(f"({a} {op_str} {b})")
                        else:
                            stack.append(f"{op_str}({a}, {b})")
                    else:
                        stack.append(f"{op_str}(?, ?)")
    
    return stack[0] if stack else "Invalid"

def generate_sample_data(n_samples=300, n_features=4, n_stocks=20, seed=42):
    np.random.seed(seed)
    
    print(f"🔧 Generating multi-stock data...")
    print(f"   Time length: {n_samples}, Stocks: {n_stocks}, Features: {n_features}")
    
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    stock_codes = [f'stock_{i:03d}' for i in range(n_stocks)]

    index = pd.MultiIndex.from_product([dates, stock_codes], names=['date', 'code'])
    total_rows = len(index)
    
    data = {}
    for i in range(n_features):
        data[f'x{i}'] = np.random.randn(total_rows).cumsum() * 0.01 + np.random.randn(total_rows) * 0.1

    target = (0.3 * data['x0'] + 
              0.2 * data['x1'] + 
              -0.1 * data['x2'] if n_features > 2 else 0 + 
              np.random.randn(total_rows) * 0.2)
    
    data['target'] = target
 
    df = pd.DataFrame(data, index=index)
    
    print(f"   Data shape: {df.shape}")
    print(f"   Target-x0 correlation: {df['target'].corr(df['x0']):.4f}")
    print()
    
    return df

def analyze_factor_quality(env) -> Dict[str, Any]:
    analysis = {
        'num_factors': len(env.factor_pool),
        'target_factors': env.num_factors,
        'avg_abs_ic': 0.0,
        'max_abs_ic': 0.0,
        'min_abs_ic': 0.0,
        'ic_std': 0.0,
        'factors': []
    }
    
    if not env.factor_pool_ics:
        return analysis
    
    ics = env.factor_pool_ics
    abs_ics = [abs(ic) for ic in ics]
    
    analysis['avg_abs_ic'] = float(np.mean(abs_ics))
    analysis['max_abs_ic'] = float(np.max(abs_ics))
    analysis['min_abs_ic'] = float(np.min(abs_ics))
    analysis['ic_std'] = float(np.std(abs_ics))
    
    for i, (ic, expression) in enumerate(zip(ics, env.factor_pool_expressions)):
        try:
            infix_expr = convert_postfix_to_infix(expression)
        except:
            infix_expr = f"Factor_{i}"
        
        analysis['factors'].append({
            'factor_id': i,
            'ic': float(ic),
            'abs_ic': float(abs(ic)),
            'expression': infix_expr
        })
    
    analysis['factors'].sort(key=lambda x: x['abs_ic'], reverse=True)
    
    return analysis
