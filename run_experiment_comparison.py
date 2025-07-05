"""
Script to automate running all experiments in both modes and export metrics for hierarchy comparison tables.
"""
import os
import subprocess

def run_pipeline(mode, experiment):
    # Map experiment name for each mode
    if mode == 'single' and experiment == 'table':
        experiment = 'table_generation'
    elif mode != 'single' and experiment == 'table_generation':
        experiment = 'table'
    if mode == 'single':
        cmd = f"python single_agent_pipeline.py --experiment {experiment} --output-dir results"
    else:
        cmd = f"python pipeline.py --experiment {experiment} --output-dir results"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def rename_metrics_files(mode, experiment):
    # Move/rename metrics files for aggregation
    mapping = {
        'qa': 'qa_metrics.json',
        'summarization': 'summarization_metrics.json',
        'table_generation': 'table_metrics.json',
    }
    for task, fname in mapping.items():
        src = os.path.join('results', fname)
        if os.path.exists(src):
            dst = os.path.join('results', f'{mode}_{fname}')
            os.replace(src, dst)
            print(f"Renamed {src} -> {dst}")

def main():
    experiments = ['summarization', 'qa', 'table']
    # 先跑 manager-executor
    modes = [
        ('Manager–Executor with MCP', 'mcp', 'pipeline.py'),
        ('Executor-Only (Single-Agent)', 'single', 'single_agent_pipeline.py'),
    ]
    for mode_name, mode_prefix, script in modes:
        print(f"\n=== Running all experiments for mode: {mode_name} ===")
        for exp in experiments:
            print(f"\n--- Running {exp} experiment in mode: {mode_name} ---")
            run_pipeline(mode_prefix, exp)
            rename_metrics_files(mode_prefix, exp if exp != 'table' else 'table_generation')
    print("\nAll experiments complete. You can now run utils/heirarchy_metrics_export.py to generate the comparison table.")

if __name__ == '__main__':
    main()
