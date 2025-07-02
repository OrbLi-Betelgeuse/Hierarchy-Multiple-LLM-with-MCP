# pudge_experiment.py
'''这个代码纯是为了测试pudge的性能，放在experiments文件夹中'''
import asyncio
import logging
from typing import List, Dict, Any
from models.llm_interface import create_llm_interface

logger = logging.getLogger(__name__)


class PudgeExperiment:
    """Standalone experiment to test Pudge model as decomposer."""

    def __init__(self, pudge_config: Dict[str, Any]):
        self.pudge_config = pudge_config
        self.llm = None

        # 示例任务
        self.sample_inputs = [
            "请帮助我分析中国人口老龄化带来的社会问题，并提出应对策略。",
            "我想了解全球变暖的原因、影响及可行的应对措施。",
        ]

    async def setup(self):
        self.llm = create_llm_interface(
            provider=self.pudge_config["provider"],
            model_name=self.pudge_config["model"],
            **self.pudge_config.get("kwargs", {})
        )
        logger.info("Pudge LLM 初始化完成")
        print("Pudge LLM 初始化完成")

    async def run(self) -> List[List[str]]:
        """运行任务分解测试"""
        all_results = []
        for text in self.sample_inputs:
            prompt = f"请将以下任务拆解成三个步骤，用JSON数组返回：{text}"
            response = await self.llm.generate(prompt)
            try:
                steps = response.content
                logger.info(f"原始任务: {text}\n步骤: {steps}")
                print(f"原始任务: {text}\n步骤: {steps}")
                all_results.append(steps)
            except Exception as e:
                logger.error(f"解析失败: {e}")
                print(f"解析失败: {e}")
        return all_results
