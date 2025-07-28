#!/usr/bin/env python3
"""
Generate a concise summary table for model size comparison
"""

import json
import os


def generate_summary_table():
    """Generate a concise summary table focusing on key metrics"""

    # Load the detailed comparison data
    with open("results/model_size_comparison.json", "r") as f:
        data = json.load(f)

    # Create summary table
    summary = []

    for task_data in data:
        task = task_data["Task"]

        # Skip tasks with errors
        if "1.5B Status" in task_data and task_data["1.5B Status"] == "Error":
            continue

        summary_row = {
            "Task": task,
            "1.5B Success Rate": task_data.get("1.5B Success Rate", "N/A"),
            "1.5B Avg Time": task_data.get("1.5B Avg Time", "N/A"),
            "1.5B Memory (MB)": task_data.get("1.5B Memory (MB)", "N/A"),
            "7B Success Rate": task_data.get("7B Success Rate", "N/A"),
            "7B Avg Time": task_data.get("7B Avg Time", "N/A"),
            "7B Memory (MB)": task_data.get("7B Memory (MB)", "N/A"),
            "14B Success Rate": task_data.get("14B Success Rate", "N/A"),
            "14B Avg Time": task_data.get("14B Avg Time", "N/A"),
            "14B Memory (MB)": task_data.get("14B Memory (MB)", "N/A"),
        }

        summary.append(summary_row)

    # Save summary
    with open("results/model_size_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate markdown summary
    markdown = "# Model Size Impact Analysis - Summary\n\n"
    markdown += "## Key Findings\n\n"

    if summary:
        # Table header
        markdown += "| Task | 1.5B Success | 1.5B Time | 1.5B Memory | 7B Success | 7B Time | 7B Memory | 14B Success | 14B Time | 14B Memory |\n"
        markdown += "|------|-------------|-----------|-------------|------------|---------|-----------|-------------|----------|------------|\n"

        for row in summary:
            markdown += f"| {row['Task']} | {row['1.5B Success Rate']} | {row['1.5B Avg Time']} | {row['1.5B Memory (MB)']} | {row['7B Success Rate']} | {row['7B Avg Time']} | {row['7B Memory (MB)']} | {row['14B Success Rate']} | {row['14B Avg Time']} | {row['14B Memory (MB)']} |\n"

    # Add analysis
    markdown += "\n## Analysis\n\n"

    if summary:
        # Analyze performance trends
        markdown += "### Performance Trends:\n\n"

        for row in summary:
            task = row["Task"]
            markdown += f"**{task.upper()}:**\n"

            # Compare success rates
            success_1_5b = row["1.5B Success Rate"]
            success_7b = row["7B Success Rate"]
            success_14b = row["14B Success Rate"]

            if success_1_5b != "N/A" and success_7b != "N/A" and success_14b != "N/A":
                markdown += f"- Success Rate: {success_1_5b} (1.5B) → {success_7b} (7B) → {success_14b} (14B)\n"

            # Compare execution times
            time_1_5b = row["1.5B Avg Time"]
            time_7b = row["7B Avg Time"]
            time_14b = row["14B Avg Time"]

            if time_1_5b != "N/A" and time_7b != "N/A" and time_14b != "N/A":
                markdown += f"- Execution Time: {time_1_5b} (1.5B) → {time_7b} (7B) → {time_14b} (14B)\n"

            # Compare memory usage
            mem_1_5b = row["1.5B Memory (MB)"]
            mem_7b = row["7B Memory (MB)"]
            mem_14b = row["14B Memory (MB)"]

            if mem_1_5b != "N/A" and mem_7b != "N/A" and mem_14b != "N/A":
                markdown += f"- Memory Usage: {mem_1_5b}MB (1.5B) → {mem_7b}MB (7B) → {mem_14b}MB (14B)\n"

            markdown += "\n"

    # Save markdown
    with open("results/model_size_summary.md", "w") as f:
        f.write(markdown)

    print("✅ Summary files generated:")
    print("  - results/model_size_summary.json")
    print("  - results/model_size_summary.md")

    # Print summary to console
    print("\n" + markdown)


if __name__ == "__main__":
    generate_summary_table()
