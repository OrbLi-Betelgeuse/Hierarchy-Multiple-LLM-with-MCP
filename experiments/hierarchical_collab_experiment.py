"""
Hierarchical Collaboration Experiment (Experiment B)

Compares single-level (flat) vs. multi-level (hierarchical) task decomposition and coordination
on Summarization (ROUGE-L), QA (F1 Score), and Table Generation (Accuracy).
"""

import asyncio
from models.manager import Manager
from models.executor import Executor
from models.llm_interface import create_llm_interface
from models.hierarchical_coordination import HierarchicalCoordinator, HierarchyLevel
from utils.evaluation import Evaluator
from experiments.summarization_experiment import SummarizationExperiment
from experiments.qa_experiment import QAExperiment
from experiments.table_generation_experiment import TableGenerationExperiment

import logging

logging.basicConfig(level=logging.INFO)


def get_sample_summarization_task():
    return {
        "id": "doc_1",
        "title": "Artificial Intelligence in Healthcare",
        "content": """
        Artificial Intelligence (AI) has emerged as a transformative force in healthcare, 
        offering unprecedented opportunities to improve patient outcomes, enhance diagnostic 
        accuracy, and streamline healthcare delivery. This comprehensive analysis explores 
        the current state of AI applications in healthcare, examining both the remarkable 
        advances and the significant challenges that lie ahead.
        """,
        "expected_summary": "AI is transforming healthcare through improved diagnostics and patient care.",
    }


def get_sample_qa_task():
    return {
        "context": "Artificial Intelligence (AI) is transforming healthcare through improved diagnostics and patient care.",
        "questions": [
            "What is AI doing in healthcare?",
            "What are the main benefits?",
            "Are there any challenges?",
        ],
        "expected_answers": [
            "AI is transforming healthcare through improved diagnostics and patient care",
            "Improved diagnostics and better patient care",
            "Yes, there are challenges in implementation and validation",
        ],
    }


def get_sample_table_task():
    return {
        "input_text": "Laptop ($999), Phone ($699), Tablet ($399)",
        "required_columns": ["Product", "Price"],
        "format_type": "markdown",
        "expected_table": "| Product | Price |\n|---------|-------|\n| Laptop | $999 |\n| Phone | $699 |\n| Tablet | $399 |",
    }


async def run_single_level(task_type, manager_config, executor_configs):
    if task_type == "summarization":
        experiment = SummarizationExperiment(manager_config, executor_configs)
    elif task_type == "qa":
        experiment = QAExperiment(manager_config, executor_configs)
    elif task_type == "table_generation":
        experiment = TableGenerationExperiment(manager_config, executor_configs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    await experiment.setup()
    results = await experiment.run_experiment()
    return results


async def run_multi_level(task_type, manager_config, executor_configs):
    from models.mcp_protocol import MCPProtocol, Task

    mcp_protocol = MCPProtocol()
    coordinator = HierarchicalCoordinator(mcp_protocol)
    # Register nodes: root manager, middle manager, leaf executors
    await coordinator.register_node("root_manager", HierarchyLevel.ROOT, [task_type])
    await coordinator.register_node(
        "middle_manager", HierarchyLevel.MIDDLE, [task_type], parent_id="root_manager"
    )
    leaf_executors = []
    for i, cfg in enumerate(executor_configs):
        node_id = f"executor_{i+1}"
        await coordinator.register_node(
            node_id,
            HierarchyLevel.LEAF,
            cfg["capabilities"],
            parent_id="middle_manager",
        )
        # Create executor instance
        llm = create_llm_interface(
            provider=cfg["provider"], model_name=cfg["model"], **cfg["kwargs"]
        )
        leaf_executors.append(
            Executor(
                executor_id=node_id, llm_interface=llm, capabilities=cfg["capabilities"]
            )
        )
    evaluator = Evaluator()
    if task_type == "summarization":
        sample = get_sample_summarization_task()
        # Decompose at root
        request_id = await coordinator.request_decomposition(
            sample["content"], 8.0, ["summarization"]
        )
        # For demo, create 2 subtasks manually (simulate multi-level)
        mid = len(sample["content"]) // 2
        subtasks = [
            Task(
                task_id="sub1",
                task_type="summarization",
                description=sample["content"][:mid],
                parameters={},
                priority=1,
                dependencies=[],
            ),
            Task(
                task_id="sub2",
                task_type="summarization",
                description=sample["content"][mid:],
                parameters={},
                priority=1,
                dependencies=[],
            ),
        ]
        # Assign and execute subtasks
        summaries = []
        for i, subtask in enumerate(subtasks):
            result = await leaf_executors[i % len(leaf_executors)].execute_task(subtask)
            summaries.append(result.result.get("output", ""))
        # Aggregate
        final_summary = " ".join(summaries)
        # Evaluate
        quality_score = evaluator.calculate_quality_metrics(
            [{"quality_score": 0.0 if not final_summary else 0.1}]  # Placeholder
        )
        return {"summary": final_summary, "quality_score": quality_score}
    elif task_type == "qa":
        sample = get_sample_qa_task()
        # Decompose: assign each question as a subtask
        request_id = await coordinator.request_decomposition(
            sample["context"], 6.0, ["qa"]
        )
        subtasks = [
            Task(
                task_id=f"q{i+1}",
                task_type="qa",
                description=f"Context: {sample['context']}\nQuestion: {q}",
                parameters={},
                priority=1,
                dependencies=[],
            )
            for i, q in enumerate(sample["questions"])
        ]
        answers = []
        for i, subtask in enumerate(subtasks):
            result = await leaf_executors[i % len(leaf_executors)].execute_task(subtask)
            answers.append(result.result.get("output", ""))
        # Evaluate F1 (placeholder: 1 if all answers match expected, else 0)
        f1 = 1.0 if answers == sample["expected_answers"] else 0.0
        return {"answers": answers, "f1": f1}
    elif task_type == "table_generation":
        sample = get_sample_table_task()
        # Decompose: assign each row as a subtask (simulate)
        request_id = await coordinator.request_decomposition(
            sample["input_text"], 5.0, ["table_generation"]
        )
        items = [item.strip() for item in sample["input_text"].split(",")]
        subtasks = [
            Task(
                task_id=f"row{i+1}",
                task_type="table_generation",
                description=item,
                parameters={},
                priority=1,
                dependencies=[],
            )
            for i, item in enumerate(items)
        ]
        rows = []
        for i, subtask in enumerate(subtasks):
            result = await leaf_executors[i % len(leaf_executors)].execute_task(subtask)
            rows.append(result.result.get("output", ""))
        # Aggregate rows into table (placeholder: just join rows)
        generated_table = "\n".join(rows)
        # Evaluate accuracy (placeholder: 1 if matches expected, else 0)
        accuracy = (
            1.0 if generated_table.strip() == sample["expected_table"].strip() else 0.0
        )
        return {"table": generated_table, "accuracy": accuracy}
    return [{"task_id": "demo", "status": "not_implemented"}]


def compare_and_print(single_results, multi_results, task_type):
    print(f"\n=== {task_type.upper()} ===")
    print(f"Single-level results: {single_results}")
    print(f"Multi-level (hierarchical) results: {multi_results}")
    # TODO: Add metric extraction and comparison (ROUGE-L, F1, accuracy, etc.)


async def main():
    manager_config = {
        "provider": "ollama",
        "model": "qwen2.5:7b",
        "manager_id": "manager_01",
        "kwargs": {"base_url": "http://localhost:11434"},
    }
    executor_configs = [
        {
            "provider": "ollama",
            "model": "qwen2.5:7b",
            "executor_id": f"executor_{i+1}",
            "capabilities": ["summarization", "qa", "table_generation"],
            "kwargs": {"base_url": "http://localhost:11434"},
        }
        for i in range(2)
    ]
    for task_type in ["summarization", "qa", "table_generation"]:
        single_results = await run_single_level(
            task_type, manager_config, executor_configs
        )
        multi_results = await run_multi_level(
            task_type, manager_config, executor_configs
        )
        compare_and_print(single_results, multi_results, task_type)


if __name__ == "__main__":
    asyncio.run(main())
