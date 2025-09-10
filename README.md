
# RLFactor Framework

A simple reinforcement learning framework for factor mining in quantitative investment, allowing direct use of CSV data files without qlib configuration requirements.

## 🎯 Overview

This framework uses PPO (Proximal Policy Optimization) to discover effective financial factors on multi-stock panel data. Unlike AlphaGen, this lightweight framework supports direct CSV data input without complex qlib setup, making it accessible for custom datasets.

**Key Differentiators:**
- 📁 **Direct CSV Support**: No qlib configuration needed - use your own data files
- 🎲 **Sample Data Included**: Comes with random data generator for quick testing
- 🔧 **Extensible Design**: Pandas-based operator library with high scalability
- 📊 **Flexible Rewards**: IC-based rewards with easy extension to in-sample/out-of-sample IC, factor diversity control

### Key Features

- 🤖 **Automated Factor Generation**: Uses RL to build factor expressions
- 📊 **Multi-Stock Support**: Panel data with grouping operations
- 📈 **Quality Assessment**: IC-based factor evaluation with overfitting control
- 🎛️ **Flexible Configuration**: Customizable environment and training settings
- 💼 **Production Ready**: Successfully tested on real market data

## 🏆 Performance Results

The framework has been successfully deployed on real market data:
- **Asset**: Steel Rebar Futures (minute-level OHLC data)
- **Achievement**: Consistent **out-of-sample IC 0.09+** with only 10-factor alpha pool
- **Stability**: Robust performance across different market conditions

## Quick Start

```bash
python main.py
```

## Group ID Usage
For Stock Data (Multi-Asset):
# Multi-stock data with MultiIndex
```python
group_id = get_group_id(df)  # Uses level=1 (stock codes)
```
For Single-Asset High-Frequency Data:
# Generate group_id for proper groupby operations
# Example: group by trading sessions, hours, etc.
```python
group_id = df.index.hour  # or custom grouping logic
```
For Simple Time Series:
# No grouping needed
```python
group_id = None
```

