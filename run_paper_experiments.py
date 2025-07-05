"""
Script to automate running all experiments in both modes and export metrics for paper tables.
"""
import os
import subprocess

def run_pipeline(mode, experiment):
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
    experiments = ['qa', 'summarization', 'table_generation']
    for mode in ['single', 'mcp']:
        for exp in experiments:
            run_pipeline(mode, exp)
            rename_metrics_files(mode, exp)
    print("All experiments complete. You can now run utils/heirarchy_metrics_export.py to generate the paper table.")

if __name__ == '__main__':
    main()
