from __future__ import annotations

from tools.base import BaseTool

TOOL_NAME = "Generalist_Solution_Generator_Tool"

TOOL_DESCRIPTION = """
A generalized tool that takes query from the user, and answers the question step by step to the best of its ability.
"""

TOOL_DEMO_COMMANDS = {
    "command": 'execution = tool.execute(query="Summarize the following text in a few lines")',
    "description": "Generate a short summary given the query from the user.",
}


class Base_Generator_Tool(BaseTool):
    def __init__(self, llm_engine):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=TOOL_DESCRIPTION,
            demo_commands=TOOL_DEMO_COMMANDS

        )
        self.llm_engine = llm_engine
        self.last_generation_turn = None

    async def execute(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]
        out = await self.llm_engine.generate(messages)
        self.last_generation_turn = {
            "source": "base_generator",
            "tokens": list(out.prompt_token_ids) + list(out.token_ids),
            "token_ids": list(out.token_ids),
            "response_length": len(out.token_ids),
            "loss_mask": [1] * len(out.token_ids),
            "rollout_log_probs": list(out.log_probs),
            "log_probs": list(out.log_probs),
            "prompt": out.prompt_text,
            "response": out.response,
        }
        return out.response
