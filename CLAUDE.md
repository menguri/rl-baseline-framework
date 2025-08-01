# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a modular reinforcement learning framework supporting PPO, TRPO, and DDPG algorithms on CartPole and LunarLanderContinuous environments. The framework uses PyTorch and integrates with Weights & Biases for experiment tracking.

## Common Commands

### Running Experiments
```bash
# Install dependencies
pip install -r requirements.txt

# Single experiment with PPO on CartPole
python -m main train --config config/cartpole_ppo.yaml

# Multi-seed experiment (recommended for reliable results)
python -m main multi --config config/cartpole_ppo_wandb.yaml --seeds 0 1 2 3 4

# Run overnight experiments (automated multi-seed runs)
bash scripts/run_overnight_experiments.sh

# Simulate trained models
bash scripts/simulate_lunarlander.sh
```

### Visualization and Analysis
```bash
# Plot episode-based learning curves
python -m main plot --results_dir results/cartpole_ppo --plot_type learning_curves

# Plot step-based learning curves (for same-step comparisons)
python -m main plot --results_dir results/cartpole_ppo --plot_type step_learning_curves

# Compare multiple algorithms (episode-based)
python -m main plot --plot_type comparison --comparison_dirs results/cartpole_ppo results/cartpole_trpo --labels PPO TRPO

# Compare multiple algorithms (step-based - for fair comparison)
python -m main plot --plot_type step_comparison --comparison_dirs results/cartpole_ppo results/cartpole_trpo --labels PPO TRPO
```

### Experiment Tracking Setup
```bash
# Required for wandb integration
wandb login
wandb status  # verify login status
```

## Architecture

### Core Components
- **algorithms/**: Base classes and concrete implementations (PPO, TRPO, DDPG)
  - `BaseAlgorithm`: Abstract base class for all RL algorithms
  - `BaseOnPolicyAlgorithm`: Base for policy gradient methods (PPO, TRPO)  
  - `BaseOffPolicyAlgorithm`: Base for off-policy methods (DDPG)

- **environments/**: Environment wrappers extending gymnasium environments
  - `CartPoleEnv`: Discrete action space environment
  - `LunarLanderContinuousEnv`: Continuous action space environment

- **networks/**: Neural network architectures
  - `MLP`: Multi-layer perceptron for value functions
  - `Actor`/`Critic`: Policy and value networks for on-policy algorithms
  - `DDPGActor`/`DDPGCritic`: Specialized networks for DDPG

- **train/**: Training orchestration
  - `Trainer`: Main training loop coordination
  - `MultiSeedTrainer`: Manages multiple seed experiments
  - `run_experiment.py`: Entry point for single experiments

### Configuration System
All experiments use YAML configuration files in `config/`. Key sections:
- `environment`: Environment name, seed
- `algorithm`: Hyperparameters (learning rates, gamma, clip_ratio, etc.)
- `network`: Architecture (hidden_dims, activation)
- `training`: Episode limits, intervals, logging
- `logging`: wandb project settings, tensorboard options

### Results Structure
Results are saved in `results/[experiment_name]/`:
- `multi_seed_results.json`: Aggregated statistics across seeds
- `seed_X/`: Individual seed results with models, metrics, logs
- `wandb/`: Weights & Biases run data

## Development Notes

### Adding New Algorithms
1. Create class inheriting from appropriate base in `algorithms/`
2. Implement abstract methods: `select_action`, `update`, `save`, `load`
3. Add algorithm import and instantiation logic in `train/trainer.py`
4. Create configuration YAML file

### Adding New Environments
1. Create wrapper class in `environments/` extending gymnasium environment
2. Add environment instantiation logic in `train/trainer.py:_setup_environment`
3. Create appropriate configuration files

### Multi-seed Experiments
The framework automatically runs multiple seeds and computes statistics. Use `--num_workers` to control parallelization (currently set to 1 to avoid resource conflicts).

### Wandb Integration
Set `use_wandb: true` in config files and ensure proper `wandb_project` and `wandb_entity` settings. The framework logs real-time metrics including episode rewards, losses, and evaluation results.

### Step-based Logging and Same-step Algorithm Comparison
The framework now supports step-based logging alongside episode-based logging for fair algorithm comparisons:

**Configuration**:
```yaml
logging:
  enable_step_logging: true
  step_log_interval: 1000  # IMPORTANT: Use same interval across all algorithms for fair comparison
```

**⚠️ Critical Note**: All algorithms must use the **same `step_log_interval`** (e.g., 1000) to ensure fair comparison in wandb and plotting. Different intervals will result in misaligned data points and invalid comparisons.

**Benefits**:
- Compare algorithms at identical training steps rather than episodes
- Account for different episode lengths across algorithms  
- Better sample efficiency analysis
- True convergence comparison independent of episode structure

**Usage**:
- Use `step_learning_curves` and `step_comparison` plot types
- Step-based metrics are stored in `metrics.json` under `step_metrics` section
- Multi-seed aggregation includes `step_based_summary` in results files