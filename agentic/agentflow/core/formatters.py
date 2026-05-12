import re


def extract_context_subgoal_and_tool(raw_response: str) -> tuple[str, str, str]:
    """
    Extract Context / Sub-Goal / Tool Name from the raw LLM response.
    Compatible with Markdown bold markers and multi-line content; expected format:
      **Justification:** ...
      **Context:** ...
      **Sub-Goal:** ...
      **Tool Name:** ...
    """

    text = raw_response

    # Keep only the last assistant segment (guard against leftover system/user template text)
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]

    # Strip <think> ... </think> blocks, keeping only the formal answer
    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    # Remove Markdown bold markers to simplify pattern matching
    text = text.replace("**", "")

    # Match:
    #   Context:   <any content>
    #   Sub-Goal: <any content>
    #   Tool Name:<any content>
    pattern = r"Context:\s*(.*?)Sub-Goal:\s*(.*?)Tool Name:\s*(.*?)\s*(?:```)?\s*(?=\n\n|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return "", "", ""

    # Use the last match (typically the result of the current step)
    context, sub_goal, tool_name = matches[-1]
    context = context.strip()
    sub_goal = sub_goal.strip()
    # Strip surrounding whitespace and trim trailing special tokens (e.g. <|im_end|>)
    tool_name = tool_name.strip()
    if "<|im_end|>" in tool_name:
        tool_name = tool_name.split("<|im_end|>", 1)[0].strip()
    return context, sub_goal, tool_name


def extract_context_subgoal_and_tool_v2(raw_response: str) -> tuple[str, str, str]:
    """
    Extract Context / Sub-Goal / Tool Name from the raw LLM response.
    Compatible with XML output format:
      <context>...</context>
      <sub_goal>...</sub_goal>
      <tool_name>...</tool_name>
    """

    text = raw_response

    # Keep only the last assistant segment (guard against leftover system/user template text)
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]

    # Strip <think> ... </think> blocks, keeping only the formal answer
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]

    # Use regex to extract content within tags (DOTALL allows matching across multiple lines)
    context_matches = re.findall(r"<context>(.*?)</context>", text, re.DOTALL | re.IGNORECASE)
    sub_goal_matches = re.findall(r"<sub_goal>(.*?)</sub_goal>", text, re.DOTALL | re.IGNORECASE)
    tool_name_matches = re.findall(r"<tool_name>(.*?)</tool_name>", text, re.DOTALL | re.IGNORECASE)

    # Use the last match if multiple exist (to catch the final thought/result)
    context = context_matches[-1].strip() if context_matches else ""
    sub_goal = sub_goal_matches[-1].strip() if sub_goal_matches else ""
    tool_name = tool_name_matches[-1].strip() if tool_name_matches else ""

    # Strip trailing special tokens just in case the model forgets the closing tag 
    # or generates tokens right next to the value
    if "<|im_end|>" in tool_name:
        tool_name = tool_name.replace("<|im_end|>", "").strip()
        
    return context, sub_goal, tool_name