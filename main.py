import numpy as np
import pandas as pd
import random

from env import FactorEnvironment
from operators import unary_rolling, binary_rolling, operater_dict
from utils import generate_sample_data, analyze_factor_quality
from agent import train_factor_agent, test_trained_agent

def test_environment_basic():
    df = generate_sample_data(n_samples=200, n_features=10,n_stocks=10)
    features = [col for col in df.columns if col.startswith('x')]
    target = 'target'
    
    print(f"Data info:")
    print(f"   Shape: {df.shape}")
    print(f"   Features: {features}")
    print(f"   Target: {target}")

    env = FactorEnvironment(
        features=features,
        unary_ops=unary_rolling,
        binary_ops=binary_rolling,
        operator_dict=operater_dict,
        df=df,
        target=target,
        num_factors=5,
        max_expr_length=8,
        min_expr_length=2,
        window_sizes=[5, 10],
        verbose=True,
        group_id=None
    )

    print(f"   State dim: {env.state_dim}")
    print(f"   Action dim: {env.action_dim}")
    print(f"   Target factors: {env.num_factors}")
    print()
    
    return env

def run_random_policy(env, max_steps=200):
    env.reset()
    total_reward = 0
    steps = 0
    factors_created = 0
    
    print("Random exploration...")
    while not env.episode_done and steps < max_steps:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            print("   No valid actions, exiting")
            break
    
        action_idx = random.choice(valid_actions)

        state, reward, done, info = env.step(action_idx)
        total_reward += reward
        steps += 1

        if info.get('factor_done', False):
            factors_created += 1
            print(f"   Created factor {factors_created} (reward: {reward:.4f})")

        if steps >= max_steps:
            print(f"   Reached max steps {max_steps}")
            break
    
    print(f"\nRandom policy results:")
    print(f"   Total steps: {steps}")
    print(f"   Total reward: {total_reward:.4f}")
    print(f"   Factors created: {len(env.factor_pool)}")
    print()
    
    return env

def run_ppo_training(env_config):
    df = generate_sample_data(n_samples=300, n_features=3, seed=123)
    features = [col for col in df.columns if col.startswith('x')]
    
    train_env = FactorEnvironment(
        features=features,
        unary_ops=unary_rolling,
        binary_ops=binary_rolling,
        operator_dict=operater_dict,
        df=df,
        target='target',
        num_factors=10,
        max_expr_length=8,
        min_expr_length=2,
        window_sizes=[5, 10, 15],
        verbose=False,
        group_id=None
    )
    
    print("Starting PPO training...")
    trained_ppo = train_factor_agent(
        train_env, 
        max_episodes=50, 
        update_frequency=200,
        verbose=True
    )
    
    print("\nTesting trained agent...")
    summary = test_trained_agent(train_env, trained_ppo, num_test_episodes=3)
    
    print("PPO training complete!")
    return train_env, trained_ppo, summary

def compare_strategies():
    print("\nStrategy comparison")
    print("=" * 50)
    
    print("1. Random policy results:")
    random_env = test_environment_basic()
    random_env = run_random_policy(random_env, max_steps=150)
    random_quality = analyze_factor_quality(random_env)
    
    print("2. PPO policy results:")
    ppo_env, ppo_agent, ppo_summary = run_ppo_training({})
    ppo_quality = analyze_factor_quality(ppo_env)
    
    print("Comparison:")
    print(f"   Random policy:")
    print(f"     Factors: {len(random_env.factor_pool)}")
    print(f"     Avg |IC|: {random_quality.get('avg_abs_ic', 0):.5f}")
    print(f"     Max |IC|: {random_quality.get('max_abs_ic', 0):.5f}")
    
    print(f"   PPO policy:")
    print(f"     Factors: {len(ppo_env.factor_pool)}")
    print(f"     Avg |IC|: {ppo_quality.get('avg_abs_ic', 0):.5f}")
    print(f"     Max |IC|: {ppo_quality.get('max_abs_ic', 0):.5f}")
    
    if random_quality.get('avg_abs_ic', 0) > 0:
        improvement = (ppo_quality.get('avg_abs_ic', 0) - random_quality.get('avg_abs_ic', 0)) / random_quality.get('avg_abs_ic', 0) * 100
        print(f"   PPO vs Random avg |IC| improvement: {improvement:+.1f}%")
    
    return random_env, ppo_env, ppo_agent

def demonstrate_best_factors(env):
    if not env.factor_pool:
        print("   No factors generated")
        return

    ics = [abs(ic) for ic in env.factor_pool_ics]
    sorted_indices = np.argsort(ics)[::-1]
    
    print("Top 5 factors:")
    for i, idx in enumerate(sorted_indices[:5]):
        if idx < len(env.factor_pool_expressions):
            try:
                from env import convert_postfix_to_infix
                expr = convert_postfix_to_infix(env.factor_pool_expressions[idx])
            except:
                expr = f"Factor_{idx}"
            ic_value = env.factor_pool_ics[idx]
            print(f"   {i+1}. IC={ic_value:8.5f} | {expr}")
    
    print()

def main():
    try:
        env = test_environment_basic()
        random_env = run_random_policy(env, max_steps=150)
        random_summary = random_env.generate_summary()
        demonstrate_best_factors(random_env)
        ppo_env, ppo_agent, ppo_summary = run_ppo_training({})
        demonstrate_best_factors(ppo_env)
        random_quality = analyze_factor_quality(random_env)
        ppo_quality = analyze_factor_quality(ppo_env)
        
        print("Final comparison:")
        print(f"   Random: {len(random_env.factor_pool)} factors, avg |IC|: {random_quality.get('avg_abs_ic', 0):.5f}")
        print(f"   PPO:  {len(ppo_env.factor_pool)} factors, avg |IC|: {ppo_quality.get('avg_abs_ic', 0):.5f}")
        
        return {
            'random_env': random_env,
            'ppo_env': ppo_env,
            'ppo_agent': ppo_agent,
            'random_summary': random_summary,
            'ppo_summary': ppo_summary
        }
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
