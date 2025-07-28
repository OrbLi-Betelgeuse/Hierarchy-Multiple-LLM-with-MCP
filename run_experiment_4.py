#!/usr/bin/env python3
"""
Experiment 4: Model Size Impact Analysis
Test the performance impact of different executor model sizes (1.5B, 7B, 14B)
"""

import os
import json
import subprocess
import time
from pathlib import Path


def run_experiment_with_config(config_file, model_size):
    """Run experiments with a specific model size configuration"""
    print(f"\n{'='*60}")
    print(f"Running Experiment 4 with {model_size} model")
    print(f"{'='*60}")

    # Copy config to the standard config file
    subprocess.run(f"cp {config_file} config.json", shell=True, check=True)

    # Run all experiments
    experiments = ["qa", "summarization", "table"]
    results = {}

    for exp in experiments:
        print(f"\nRunning {exp} experiment with {model_size} model...")

        # Run MCP pipeline
        cmd = f"python pipeline.py --experiment {exp} --output-dir results"
        print(f"Command: {cmd}")

        start_time = time.time()
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            end_time = time.time()

            if result.returncode == 0:
                print(
                    f"✅ {exp} experiment completed successfully in {end_time - start_time:.2f}s"
                )
            else:
                print(f"❌ {exp} experiment failed")
                print(f"Error: {result.stderr}")
                continue

        except Exception as e:
            print(f"❌ Error running {exp} experiment: {e}")
            continue

    # Rename results to include model size
    rename_results(model_size)

    return True


def rename_results(model_size):
    """Rename result files to include model size"""
    result_files = [
        "results/qa_metrics.json",
        "results/summarization_metrics.json",
        "results/table_metrics.json",
    ]

    for file_path in result_files:
        if os.path.exists(file_path):
            new_name = file_path.replace(".json", f"_{model_size}.json")
            os.rename(file_path, new_name)
            print(f"Renamed {file_path} -> {new_name}")


def generate_comparison_table():
    """Generate comparison table for different model sizes"""
    print("\n" + "=" * 60)
    print("Generating Model Size Comparison Table")
    print("=" * 60)

    model_sizes = ["1.5b", "7b", "14b"]
    tasks = ["qa", "summarization", "table"]

    comparison_data = []

    for task in tasks:
        task_data = {"Task": task}

        for model_size in model_sizes:
            metrics_file = f"results/{task}_metrics_{model_size}.json"

            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)

                # Check if there's an error
                if "error" in metrics:
                    task_data[f"{model_size.upper()} Status"] = "Error"
                    continue

                # Extract key metrics
                if "success_rate" in metrics:
                    task_data[f"{model_size.upper()} Success Rate"] = (
                        f"{metrics['success_rate']:.2%}"
                    )
                if "average_execution_time" in metrics:
                    task_data[f"{model_size.upper()} Avg Time"] = (
                        f"{metrics['average_execution_time']:.4f}s"
                    )
                if "total_tasks" in metrics:
                    task_data[f"{model_size.upper()} Total Tasks"] = metrics[
                        "total_tasks"
                    ]
                if "successful_tasks" in metrics:
                    task_data[f"{model_size.upper()} Successful Tasks"] = metrics[
                        "successful_tasks"
                    ]

                # Extract resource utilization
                if "resource_utilization" in metrics:
                    res = metrics["resource_utilization"]
                    if "memory_peak_bytes" in res:
                        task_data[f"{model_size.upper()} Memory (MB)"] = (
                            f"{res['memory_peak_bytes']/1024/1024:.2f}"
                        )
                    if "cpu_user_time" in res:
                        task_data[f"{model_size.upper()} CPU Time"] = (
                            f"{res['cpu_user_time']:.3f}s"
                        )
            else:
                task_data[f"{model_size.upper()} Status"] = "File Not Found"

        comparison_data.append(task_data)

    # Save comparison table
    output_file = "results/model_size_comparison.json"
    with open(output_file, "w") as f:
        json.dump(comparison_data, f, indent=2)

    print(f"✅ Comparison table saved to {output_file}")

    # Generate markdown table
    generate_markdown_table(comparison_data)


def generate_markdown_table(data):
    """Generate markdown table from comparison data"""
    if not data:
        return

    # Get all columns
    columns = list(data[0].keys())

    # Create markdown table
    markdown = "# Model Size Impact Analysis Results\n\n"
    markdown += "| " + " | ".join(columns) + " |\n"
    markdown += "|" + "|".join(["---"] * len(columns)) + "|\n"

    for row in data:
        markdown += (
            "| " + " | ".join([str(row.get(col, "N/A")) for col in columns]) + " |\n"
        )

    # Save markdown file
    output_file = "results/model_size_comparison.md"
    with open(output_file, "w") as f:
        f.write(markdown)

    print(f"✅ Markdown table saved to {output_file}")

    # Print table to console
    print("\n" + markdown)


def main():
    """Main function to run Experiment 4"""
    print("🚀 Starting Experiment 4: Model Size Impact Analysis")
    print("Testing executor models: 1.5B, 7B, 14B")

    # Model configurations
    configs = [
        ("config_1.5b.json", "1.5b"),
        ("config_7b.json", "7b"),
        ("config_14b.json", "14b"),
    ]

    # Run experiments for each model size
    for config_file, model_size in configs:
        if not os.path.exists(config_file):
            print(f"❌ Config file {config_file} not found, skipping...")
            continue

        success = run_experiment_with_config(config_file, model_size)
        if not success:
            print(f"❌ Failed to run experiments with {model_size} model")
            continue

    # Generate comparison table
    generate_comparison_table()

    print("\n🎉 Experiment 4 completed!")
    print(
        "Check results/model_size_comparison.json and results/model_size_comparison.md for detailed results"
    )


if __name__ == "__main__":
    main()
