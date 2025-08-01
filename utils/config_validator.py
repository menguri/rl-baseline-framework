#!/usr/bin/env python3
"""
설정 파일 검증 유틸리티
"""

import os
import yaml
import glob
from pathlib import Path

def load_config(config_path):
    """YAML 설정 파일을 로드합니다."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def validate_step_logging_consistency():
    """모든 설정 파일의 스텝 로깅 일관성을 검증합니다."""
    
    print("=" * 60)
    print("스텝 로깅 설정 일관성 검증")
    print("=" * 60)
    
    # config 디렉토리의 모든 YAML 파일 찾기
    config_dir = Path(__file__).parent.parent / "config"
    config_files = glob.glob(str(config_dir / "*.yaml"))
    
    # step_logging_config.yaml은 제외 (공통 설정 파일)
    config_files = [f for f in config_files if not f.endswith("step_logging_config.yaml")]
    
    step_intervals = {}
    issues = []
    
    for config_file in config_files:
        try:
            config = load_config(config_file)
            filename = os.path.basename(config_file)
            
            if 'logging' in config and 'step_log_interval' in config['logging']:
                step_interval = config['logging']['step_log_interval']
                step_intervals[filename] = step_interval
                
                enable_step_logging = config['logging'].get('enable_step_logging', False)
                
                print(f"✓ {filename}")
                print(f"  - enable_step_logging: {enable_step_logging}")
                print(f"  - step_log_interval: {step_interval}")
                
                if not enable_step_logging:
                    issues.append(f"{filename}: enable_step_logging이 False입니다")
                    
            else:
                issues.append(f"{filename}: 스텝 로깅 설정이 없습니다")
                print(f"❌ {filename}: 스텝 로깅 설정이 없음")
                
        except Exception as e:
            issues.append(f"{filename}: 설정 파일 읽기 오류 - {e}")
            print(f"❌ {filename}: 오류 - {e}")
    
    print("\n" + "=" * 60)
    print("검증 결과")
    print("=" * 60)
    
    # 스텝 간격 일관성 검사
    if step_intervals:
        unique_intervals = set(step_intervals.values())
        
        if len(unique_intervals) == 1:
            interval = list(unique_intervals)[0]
            print(f"✅ 모든 알고리즘이 동일한 스텝 간격을 사용합니다: {interval}")
        else:
            print(f"❌ 스텝 간격이 일치하지 않습니다:")
            for filename, interval in step_intervals.items():
                print(f"  - {filename}: {interval}")
            issues.append("스텝 간격이 알고리즘별로 다릅니다")
    
    # 문제점 요약
    if issues:
        print(f"\n⚠️ 발견된 문제점들:")
        for issue in issues:
            print(f"  - {issue}")
        
        print(f"\n💡 해결 방법:")
        print(f"  1. 모든 config 파일에 다음 설정 추가:")
        print(f"     logging:")
        print(f"       enable_step_logging: true")
        print(f"       step_log_interval: 1000")
        print(f"  2. 모든 알고리즘에서 동일한 step_log_interval 사용")
        
        return False
    else:
        print(f"✅ 모든 설정이 올바릅니다!")
        return True

def suggest_step_interval(environment_name, max_episodes=None):
    """환경과 에피소드 수에 따른 권장 스텝 간격을 제안합니다."""
    
    recommendations = {
        "CartPole-v1": {
            "typical_episode_length": 200,
            "recommended_interval": 1000,
            "reason": "짧은 에피소드, 빠른 학습"
        },
        "LunarLanderContinuous-v3": {
            "typical_episode_length": 1000,
            "recommended_interval": 1000,
            "reason": "긴 에피소드, 점진적 학습"
        }
    }
    
    if environment_name in recommendations:
        rec = recommendations[environment_name]
        print(f"\n환경 '{environment_name}'에 대한 권장사항:")
        print(f"  - 일반적인 에피소드 길이: {rec['typical_episode_length']} 스텝")
        print(f"  - 권장 스텝 간격: {rec['recommended_interval']}")
        print(f"  - 이유: {rec['reason']}")
        
        if max_episodes:
            total_steps = max_episodes * rec['typical_episode_length']
            data_points = total_steps // rec['recommended_interval']
            print(f"  - 예상 총 스텝: {total_steps:,}")
            print(f"  - 예상 데이터 포인트: {data_points}")
        
        return rec['recommended_interval']
    else:
        print(f"환경 '{environment_name}'에 대한 권장사항이 없습니다.")
        print(f"일반적인 권장 간격: 1000 스텝")
        return 1000

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="설정 파일 검증")
    parser.add_argument("--validate", action="store_true", help="스텝 로깅 일관성 검증")
    parser.add_argument("--suggest", type=str, help="환경에 대한 권장 스텝 간격 제안")
    parser.add_argument("--max_episodes", type=int, help="최대 에피소드 수 (제안용)")
    
    args = parser.parse_args()
    
    if args.validate:
        validate_step_logging_consistency()
    
    if args.suggest:
        suggest_step_interval(args.suggest, args.max_episodes)
    
    if not args.validate and not args.suggest:
        print("사용법:")
        print("  python utils/config_validator.py --validate")
        print("  python utils/config_validator.py --suggest CartPole-v1 --max_episodes 1000")

if __name__ == "__main__":
    main()