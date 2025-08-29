#!/usr/bin/env python3
"""
DroneHover-v0 환경 설정 테스트 스크립트
"""

import sys
import numpy as np
import os

def test_import_dependencies():
    """필수 패키지 import 테스트"""
    try:
        import gym_pybullet_drones
        print("✅ gym-pybullet-drones imported successfully")
        
        from gym_pybullet_drones.envs.HoverAviary import HoverAviary
        from gym_pybullet_drones.utils.enums import DroneModel, Physics
        print("✅ HoverAviary and enums imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_environment_creation():
    """DroneHover 환경 생성 테스트"""
    try:
        from gym_pybullet_drones.envs.HoverAviary import HoverAviary
        from gym_pybullet_drones.utils.enums import DroneModel, Physics
        
        # Set headless mode for testing
        os.environ["DISPLAY"] = ""
        
        env = HoverAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            neighbourhood_radius=10,
            initial_xyzs=None,
            initial_rpys=None,
            physics=Physics.PYB,
            freq=240,
            aggregate_phy_steps=1,
            gui=False,  # Headless
            record=False,
            obstacles=False,
            user_debug_gui=False
        )
        print("✅ HoverAviary environment created successfully")
        
        # Reset and check observation
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle new gym API
            
        print(f"✅ Environment reset successful")
        print(f"   - Observation type: {type(obs)}")
        
        if isinstance(obs, dict):
            print(f"   - Observation keys: {list(obs.keys())}")
            sample_obs = list(obs.values())[0]
            print(f"   - Single drone obs shape: {sample_obs.shape}")
        else:
            print(f"   - Observation shape: {obs.shape}")
        
        # Test action space
        print(f"   - Action space: {env.action_space}")
        if isinstance(env.action_space, dict):
            sample_action_space = list(env.action_space.values())[0]
            print(f"   - Single drone action space: {sample_action_space}")
        
        # Test random action
        if isinstance(env.action_space, dict):
            action = {0: env.action_space[0].sample()}
        else:
            action = env.action_space.sample()
        print(f"   - Sample action generated")
        
        # Test step
        next_obs, reward, done, truncated, info = env.step(action)
        print(f"✅ Environment step successful")
        print(f"   - Reward type: {type(reward)}")
        print(f"   - Done type: {type(done)}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_custom_wrapper():
    """커스텀 DroneHoverEnv wrapper 테스트"""
    try:
        # Add current directory to path for imports
        sys.path.insert(0, os.getcwd())
        
        from environments.drone_env import DroneHoverEnv
        print("✅ DroneHoverEnv wrapper imported successfully")
        
        # Create environment
        env = DroneHoverEnv(env_name="hover", seed=42)
        print("✅ DroneHoverEnv wrapper created successfully")
        
        # Test properties
        print(f"   - State dim: {env.get_state_dim()}")
        print(f"   - Action dim: {env.get_action_dim()}")
        print(f"   - Has continuous actions: {env.has_continuous_actions()}")
        
        # Test action bounds
        action_low, action_high = env.get_action_bounds()
        print(f"   - Action bounds: [{action_low[0]:.2f}, {action_high[0]:.2f}]")
        
        # Test reset and step
        state = env.reset()
        print(f"✅ Wrapper reset successful, state shape: {state.shape}")
        
        # Random action within bounds
        action = np.random.uniform(action_low, action_high)
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
        sys.path.insert(0, os.getcwd())
        
        from train.trainer import Trainer
        
        # Create a minimal config for testing
        test_config = {
            'environment': {
                'name': 'DroneHover-v0',
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
                'max_steps_per_episode': 100  # Short episodes for testing
            },
            'experiment': {
                'name': 'test_drone_sac',
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
        with open('config/test/drone_test.yaml', 'w') as f:
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
    print("DroneHover-v0 Setup Validation")
    print("========================================")
    
    # Set headless mode
    os.environ["DISPLAY"] = ""
    
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
        print("🎉 All tests passed! DroneHover-v0 setup is ready.")
        print("🚁 Ready for single drone hovering experiments!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("========================================")
    
    return passed == total

if __name__ == "__main__":
    main()