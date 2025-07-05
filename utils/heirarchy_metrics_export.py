"""
Script to aggregate experiment results for hierarchy experiments, including relative improvement calculation.
"""
import json
import os
from typing import Dict, Any

def load_metrics(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)

def get_metric(metrics: Dict[str, Any], key: str, default=0.0):
    # 支持多层key（如 resource_utilization.memory_peak_bytes）
    if '.' in key:
        keys = key.split('.')
        val = metrics
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val
    return metrics.get(key, default)

def compute_relative_improvement(val1, val2):
    if val1 == 0:
        return 'N/A'
    return f"{((val2 - val1) / abs(val1)) * 100:.1f}%"

def main():
    # Paths to metrics files for both modes
    results_dir = 'results'
    tasks = [
        ('Summarization (Quality)', 'summarization_metrics.json', 'average_quality_score'),
        ('QA (Accuracy)', 'qa_metrics.json', 'success_rate'),
        ('Summarization Time (avg)', 'summarization_metrics.json', 'average_execution_time'),
        ('QA Time (avg)', 'qa_metrics.json', 'average_execution_time'),
        ('Table Time (avg)', 'table_metrics.json', 'average_execution_time'),
        ('Summarization Memory (peak)', 'summarization_metrics.json', 'resource_utilization.memory_peak_bytes'),
        ('QA Memory (peak)', 'qa_metrics.json', 'resource_utilization.memory_peak_bytes'),
        ('Table Memory (peak)', 'table_metrics.json', 'resource_utilization.memory_peak_bytes'),
        ('Summarization CPU (user)', 'summarization_metrics.json', 'resource_utilization.cpu_user_time'),
        ('QA CPU (user)', 'qa_metrics.json', 'resource_utilization.cpu_user_time'),
        ('Table CPU (user)', 'table_metrics.json', 'resource_utilization.cpu_user_time'),
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
    md_lines = []
    md_lines.append('| ' + ' | '.join(headers) + ' |')
    md_lines.append('|' + '---|' * len(headers))
    for row in table:
        md_lines.append('| ' + ' | '.join(str(row[h]) for h in headers) + ' |')
    md_content = '\n'.join(md_lines)
    print(md_content)
    # Save as markdown file
    with open('results/heirarchy_comparison_table.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
    # Save as json file
    with open('results/heirarchy_comparison_table.json', 'w', encoding='utf-8') as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    print("\nSaved comparison table to results/heirarchy_comparison_table.md and .json")

if __name__ == '__main__':
    main()
