"""
Script to aggregate experiment results for paper tables, including relative improvement calculation.
"""
import json
import os
from typing import Dict, Any

def load_metrics(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)

def get_metric(metrics: Dict[str, Any], key: str, default=0.0):
    return metrics.get(key, default)

def compute_relative_improvement(val1, val2):
    if val1 == 0:
        return 'N/A'
    return f"{((val2 - val1) / abs(val1)) * 100:.1f}%"

def main():
    # Paths to metrics files for both modes
    results_dir = 'results'
    tasks = [
        ('Summarization (ROUGE-L)', 'summarization_metrics.json', 'average_quality_score'),
        ('QA (F1 Score)', 'qa_metrics.json', 'average_quality_score'),
        ('Table Generation (Accuracy)', 'table_metrics.json', 'average_quality_score'),
        ('Execution Time (avg)', 'qa_metrics.json', 'average_execution_time'),
        ('Memory Usage (avg)', 'qa_metrics.json', 'average_memory_usage'),
    ]
    modes = [
        ('Executor-Only (Single-Agent)', 'single'),
        ('Manager–Executor with MCP', 'mcp'),
    ]
    # Assume files are named as <mode>_<taskfile>, e.g., single_qa_metrics.json, mcp_qa_metrics.json
    table = []
    for task_name, metric_file, metric_key in tasks:
        row = {'Task': task_name}
        vals = []
        for mode_name, mode_prefix in modes:
            fname = os.path.join(results_dir, f'{mode_prefix}_{metric_file}')
            if os.path.exists(fname):
                metrics = load_metrics(fname)
                val = get_metric(metrics, metric_key, 'N/A')
            else:
                val = 'N/A'
            row[mode_name] = val
            vals.append(val if isinstance(val, (int, float)) else 0.0)
        # Compute relative improvement
        if all(isinstance(v, (int, float)) for v in vals):
            row['Relative Improvement'] = compute_relative_improvement(vals[0], vals[1])
        else:
            row['Relative Improvement'] = 'N/A'
        table.append(row)
    # Output as markdown table
    headers = ['Task'] + [m[0] for m in modes] + ['Relative Improvement']
    print('| ' + ' | '.join(headers) + ' |')
    print('|' + '---|' * len(headers))
    for row in table:
        print('| ' + ' | '.join(str(row[h]) for h in headers) + ' |')

if __name__ == '__main__':
    main()
