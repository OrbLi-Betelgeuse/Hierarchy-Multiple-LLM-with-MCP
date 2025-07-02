'''空架子，纯为了测试ollama和其api是否正常使用'''
import asyncio
from pudge_experiment import PudgeExperiment
from task_solver_experiment import TaskSolverExperiment

pudge_cfg = {
    "provider": "ollama",
    "model": "qwen2.5:7b",
    "kwargs": {"base_url": "http://localhost:11434"},
}

executor_cfg = {
    "provider": "ollama",
    "model": "qwen2.5:14b",
    "kwargs": {"base_url": "http://localhost:11434"},
}


async def main():
    print("=== 测试 Pudge 分解能力 ===")
    pudge_exp = PudgeExperiment(pudge_cfg)
    await pudge_exp.setup()
    await pudge_exp.run()

    print("\n=== 测试 14B 子任务模型执行能力 ===")
    solver_exp = TaskSolverExperiment(executor_cfg)
    await solver_exp.setup()
    await solver_exp.run()

if __name__ == "__main__":
    asyncio.run(main())
