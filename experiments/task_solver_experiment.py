# task_solver_experiment.py
import asyncio
import logging
from typing import List, Dict, Any
from models.llm_interface import create_llm_interface

logger = logging.getLogger(__name__)


class TaskSolverExperiment:
    """Standalone experiment to test sub-task execution models."""

    def __init__(self, executor_config: Dict[str, Any]):
        self.executor_config = executor_config
        self.llm = None

        # 示例子任务（假设来自 Pudge 的分解）
        self.sample_steps = [
            "解释什么是全球变暖",
            "分析全球变暖带来的负面影响",
            "提出应对全球变暖的可行方案"
        ]

    async def setup(self):
        self.llm = create_llm_interface(
            provider=self.executor_config["provider"],
            model_name=self.executor_config["model"],
            **self.executor_config.get("kwargs", {})
        )
        logger.info("子任务模型初始化完成")
        print("子任务模型初始化完成")

    async def run(self) -> List[str]:
        """运行模型处理子任务"""
        results = []
        for step in self.sample_steps:
            # 使用 generate 或 run
            if hasattr(self.llm, "generate"):
                response = await self.llm.generate(step)
            elif hasattr(self.llm, "run"):
                response = await self.llm.run(step)
            else:
                raise AttributeError("LLM 接口没有 generate 或 run 方法")

            # 提取内容
            content = response.get("content", "") if isinstance(response, dict) else response

            logger.info(f"任务: {step}\n输出: {content}")
            print(f"任务: {step}\n输出: {content}")
            results.append(content)
        return results
