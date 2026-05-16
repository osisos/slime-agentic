import re


class Rewarder:
    def __init__(self, llm_engine):
        self.llm_engine = llm_engine

    @staticmethod
    def _score_from_response(response: str) -> float:
        match = re.search(r"VERDICT\s*:\s*(True|False)", response, re.IGNORECASE)
        if match:
            return 1.0 if match.group(1).lower() == "true" else 0.0

        match = re.search(r"<true_false>\s*:?\s*(true|false)", response, re.IGNORECASE)
        if match:
            return 1.0 if match.group(1).lower() == "true" else 0.0

        last_line = response.strip().splitlines()[-1].strip().lower() if response.strip() else ""
        if last_line in ("true", "true.", "verdict: true"):
            return 1.0
        if last_line in ("false", "false.", "verdict: false"):
            return 0.0

        return 0.0

    async def judge(self, question: str, model_response: str, groundtruth: str) -> dict:
        query_prompt = f"""You are a strict math answer evaluator.

**Task:** Read the Model Response, extract its final answer, and determine if it strictly matches the Ground Truth.

**Steps & Critical Rules:**
1. Extraction: Read the Model Response carefully. Find the final explicit answer — look for \boxed{...}, "the answer is ...", "result: ...", or the last numerical/symbolic conclusion.
2. Anti-Parroting Check (Crucial): The model MUST provide the calculated/evaluated result. If the extracted answer is merely a repetition of the target variables or expression from the question (e.g., answering "m+n+p", "x", "the area") instead of an actual value, you MUST output False.
3. NO Variable Substitution: Do NOT evaluate the model's answer on its behalf. Do NOT assume the model's variables equal the ground truth. For example, if the Ground Truth is "104" and the model outputs "m+n+p", they are NOT equivalent, even if the question asked for m+n+p. The model must explicitly output the value "104".
4. Equivalence: Compare the extracted answer to the Ground Truth. They are equivalent ONLY if they represent the exact same mathematical value or fully evaluated algebraic form independently (e.g., "1/2" == "0.5", "1000" == "1,000", but "m+n+p" != "104").
5. Strictness: If the Model Response has no clear final answer, fails the Anti-Parroting check, relies on you to substitute variables to match the truth, or does not mathematically match the Ground Truth, output False. 
6. Do NOT be lenient. When in doubt, output False.

**Inputs:**
Question: {question}

Model Response:
{model_response}

Ground Truth: {groundtruth}

**You MUST end your response with exactly one of these two lines (no extra text after it):**
VERDICT: True
VERDICT: False"""

        messages = [{"role": "user", "content": query_prompt}]
        out = await self.llm_engine.generate(messages)
        response = out.response.strip()
        score = self._score_from_response(response)

        return {
            "question": question,
            "model_response": model_response,
            "groundtruth": groundtruth,
            "prompt": query_prompt,
            "response": response,
            "score": score,
            "finish_reason": out.finish_reason,
        }

    async def compute_reward(self, question: str, model_response: str, groundtruth: str) -> float:
        judgement = await self.judge(question, model_response, groundtruth)
        return float(judgement["score"])
