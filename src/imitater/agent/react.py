import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

from .function import get_functions
from .types import Agent


if TYPE_CHECKING:
    from .function import Function


JSON_FORMAT_PROMPT = ', in a JSON format representing the kwargs (e.g. ```{"input": "hello world", "num_beams": 5}```)'


CODE_FORMAT_PROMPT = ", in triple backticks (e.g. ```code goes here```)"


REACT_PROMPT = (
    "Answer the following questions as best you can. "
    "You have access to the following tools:\n{tool_text}"
    "Use the following format to answer the question:\n"
    "```\n"
    "Thought: I need to use a tool to help me answer the question.\n"
    "Action: the action to take, should be one of [{tool_names}] if using a tool.\n"
    "Action Input: the input to the action{format_prompt}.\n"
    "```\n\n"
    "Please ALWAYS start with a Thought. "
    "If this format is used, the user will respond in the following format:\n"
    "```\n"
    "Observation: [tool response]\n"
    "```\n\n"
    "You should keep repeating the above format until you have enough information "
    "to answer the question without using any more tools. At that point, you MUST respond "
    "in the one of the following two formats:\n"
    "```\n"
    "Thought: I can answer without using any more tools.\n"
    "Answer: [your answer here]\n"
    "```\n"
    "```\n"
    "Thought: I cannot answer the question with the provided tools.\n"
    "Answer: Sorry, I cannot answer your question.\n"
    "```\n\n"
    "Below is the current conversation consisting of interleaving human and assistant messages.\n"
    "```\n"
    "{messages_text}"
    "```"
)


class ReAct(Agent):
    def __init__(self) -> None:
        self.type = "react"

    def _build_tool_text(self, functions: Dict[str, "Function"]) -> str:
        tool_text = ""
        for func in functions.values():
            param_text = ""
            for param in func.parameters:
                required = ", required" if param.name in func.required else ""
                param_text += "  - {name} ({type}{required}): {desc}\n".format(
                    name=param.name, type=param.type, required=required, desc=param.description
                )

            tool_text += "> Tool Name: {name}\nTool Description: {desc}\nTool Args:\n{args}\n".format(
                name=func.name, desc=func.description, args=param_text
            )

        return tool_text

    def _build_messages_text(self, messages: List[Dict[str, str]]) -> str:
        messages_text = ""
        for message in messages:
            if message["role"] == "user":
                messages_text += "Human: {}\n".format(message["content"])

            elif message["role"] == "assistant":
                messages_text += "Assistant: {}\n".format(message["content"])

            elif message["role"] == "tool":
                messages_text += "Observation: {}\n".format(message["content"])

            elif message["role"] == "function":
                tool_call: Dict[str, str] = json.loads(message["content"])
                messages_text += "Action: {}\nAction Input: {}\n".format(
                    tool_call.get("name", ""), tool_call.get("arguments", "")
                )

        return messages_text

    def build_prompt(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        functions = get_functions(tools)
        if "code_interpreter" in functions:
            prompt = REACT_PROMPT.format(
                tool_text=self._build_tool_text(functions),
                tool_names=", ".join(functions.keys()),
                format_prompt=CODE_FORMAT_PROMPT,
                messages_text=self._build_messages_text(messages),
            )
        else:
            prompt = REACT_PROMPT.format(
                tool_text=self._build_tool_text(functions),
                tool_names=", ".join(functions.keys()),
                format_prompt=JSON_FORMAT_PROMPT,
                messages_text=self._build_messages_text(messages),
            )

        return [{"role": "user", "content": prompt}]

    def extract_tool(self, answer: str, tools: Dict[str, Any]) -> Union[str, Tuple[str, str]]:
        functions = get_functions(tools)
        regex = re.compile(r"Action:\s*([a-zA-Z0-9_]+)\s*Action Input:\s*(.*)", re.DOTALL)
        action_match = re.search(regex, answer)
        if action_match:
            tool_name = action_match.group(1).strip()
            if tool_name not in functions:
                return "Failed: tool does not exist."

            tool_input = action_match.group(2).strip().strip('"')
            if "code_interpreter" in functions:
                arguments = {"code": tool_input}
            else:
                try:
                    arguments = json.loads(tool_input.strip("```"))
                except json.JSONDecodeError:
                    return "Failed: params not found."

            for name, value in arguments.items():
                if not functions[tool_name].verify_param(name, value):
                    return "Failed: invalid params."

            return tool_name, json.dumps(arguments, ensure_ascii=False)
        else:
            regex = re.compile(r"Answer:\s*(.*)", re.DOTALL)
            answer_match = re.search(regex, answer)
            if answer_match:
                return answer_match.group(1).strip()
            else:
                return "Failed: answers not found."

    def get_stop_word(self) -> str:
        return "Observation"


if __name__ == "__main__":
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    react = ReAct()
    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    print(react.build_prompt(messages, tools))

    answer = 'Action: get_current_weather\nAction Input: {\n"location": "Boston, MA"\n}'
    print(react.extract_tool(answer, tools))

    answer = "Thought: I can answer.\nAnswer: The weather in Boston is currently sunny."
    print(react.extract_tool(answer, tools))
