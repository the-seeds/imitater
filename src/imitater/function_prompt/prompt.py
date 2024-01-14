json_format_prompt = """in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})"""
json_format_warning_prompt = """Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}."""

REACT_PROMPT = """You are designed to help with a variety of tasks, from answering questions \
to providing summaries to other types of analyses.Answer the following questions as best you can.

## Tools
You have access to the following tools:
{tools_text}

## Output Format
To answer the question, please use the following format.

```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tools_name_text}) if using a tool.
Action Input: the input to the tool. {json_format_prompt}.
```

Please ALWAYS start with a Thought.

{json_format_warning_prompt}

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

Begin!

Question: {query}"""
