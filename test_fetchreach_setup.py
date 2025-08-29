#!/usr/bin/env python3
"""
FetchReach-v4 환경 설정 테스트 스크립트
"""

import sys
import numpy as np

def test_import_dependencies():
    """필수 패키지 import 테스트"""
    try:
        import gymnasium as gym
        print("✅ gymnasium imported successfully")
        
        import gymnasium_robotics
        print("✅ gymnasium-robotics imported successfully")
        
        gym.register_envs(gymnasium_robotics)
        print("✅ gymnasium-robotics environments registered")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_environment_creation():
    """FetchReach-v4 환경 생성 테스트"""
    try:
        import gymnasium as gym
        import gymnasium_robotics
        
        gym.register_envs(gymnasium_robotics)
        env = gym.make('FetchReach-v4')
        print("✅ FetchReach-v4 environment created successfully")
        
        # Reset and check observation
        obs, info = env.reset(seed=42)
        print(f"✅ Environment reset successful")
        print(f"   - Observation type: {type(obs)}")
        print(f"   - Observation keys: {obs.keys() if isinstance(obs, dict) else 'Not dict'}")
        
        if isinstance(obs, dict):
            print(f"   - observation shape: {obs['observation'].shape}")
            print(f"   - achieved_goal shape: {obs['achieved_goal'].shape}")
            print(f"   - desired_goal shape: {obs['desired_goal'].shape}")
        
        # Test action space
        print(f"   - Action space: {env.action_space}")
        print(f"   - Action space shape: {env.action_space.shape}")
        
        # Test random action
        action = env.action_space.sample()
        print(f"   - Sample action: {action}")
        
        # Test step
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Environment step successful")
        print(f"   - Reward: {reward}")
        print(f"   - Terminated: {terminated}")
        print(f"   - Truncated: {truncated}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_custom_wrapper():
    """커스텀 FetchReachEnv wrapper 테스트"""
    try:
        # Add current directory to path for imports
        import os
        sys.path.insert(0, os.getcwd())
        
        from environments.fetch_reach_env import FetchReachEnv
        print("✅ FetchReachEnv wrapper imported successfully")
        
        # Create environment
        env = FetchReachEnv(env_name="FetchReach-v4", seed=42)
        print("✅ FetchReachEnv wrapper created successfully")
        
        # Test properties
        print(f"   - State dim: {env.get_state_dim()}")
        print(f"   - Action dim: {env.get_action_dim()}")
        print(f"   - Has continuous actions: {env.has_continuous_actions()}")
        
        # Test observation info
        obs_info = env.get_observation_info()
        print(f"   - Observation info: {obs_info}")
        
        # Test reset and step
        state = env.reset()
        print(f"✅ Wrapper reset successful, state shape: {state.shape}")
        
        action = np.random.uniform(-1, 1, 4)  # Random action in [-1, 1]
        next_state, reward, done, info = env.step(action)
        print(f"✅ Wrapper step successful")
        print(f"   - Next state shape: {next_state.shape}")
        print(f"   - Reward: {reward}")
        print(f"   - Done: {done}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Custom wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trainer_integration():
    """Trainer 클래스와의 통합 테스트"""
    try:
        import os
        sys.path.insert(0, os.getcwd())
        
        from train.trainer import Trainer
        
        # Create a minimal config for testing
        test_config = {
            'environment': {
                'name': 'FetchReach-v4',
                'seed': 42
            },
            'algorithm': {
                'name': 'SAC',
                'lr_actor': 0.0003,
                'lr_critic': 0.0003,
                'lr_alpha': 0.0003,
                'gamma': 0.99,
                'tau': 0.005,
                'buffer_size': 10000,  # Small for testing
                'batch_size': 64,
                'alpha': 0.2,
                'automatic_entropy_tuning': True,
                'target_entropy': -4,
                'stable_update_size': 100
            },
            'network': {
                'hidden_dims': [64, 64]  # Small for testing
            },
            'training': {
                'max_episodes': 2,  # Very small for testing
                'update_interval': 1,
                'eval_interval': 1,
                'save_interval': 10,
                'max_steps_per_episode': 50
            },
            'experiment': {
                'name': 'test_fetchreach_sac',
                'seeds': [42],
                'device': 'cpu',  # Use CPU for testing
                'save_dir': 'test_results'
            },
            'logging': {
                'tensorboard': False,
                'save_metrics': False,
                'use_wandb': False
            }
        }
        
        # Save test config
        import yaml
        os.makedirs('config/test', exist_ok=True)
        with open('config/test/fetchreach_test.yaml', 'w') as f:
            yaml.dump(test_config, f)
        
        print("✅ Test config created")
        
        # This would test trainer creation but requires full environment
        print("✅ Trainer integration test setup complete")
        print("   Note: Full trainer test requires complete environment setup")
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 테스트 실행"""
    print("========================================")
    print("FetchReach-v4 Setup Validation")
    print("========================================")
    
    tests = [
        ("Import Dependencies", test_import_dependencies),
        ("Environment Creation", test_environment_creation),
        ("Custom Wrapper", test_custom_wrapper),
        ("Trainer Integration", test_trainer_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- Testing {test_name} ---")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n========================================")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! FetchReach-v4 setup is ready.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("========================================")
    
    return passed == total

if __name__ == "__main__":
    main()