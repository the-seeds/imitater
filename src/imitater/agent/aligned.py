import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

from .function import get_functions
from .types import Agent


if TYPE_CHECKING:
    from .function import Function


JSON_FORMAT_PROMPT = (
    """, in a JSON format representing the kwargs (e.g. ```{"input": "hello world", "num_beams": 5}```)"""
)


CODE_FORMAT_PROMPT = ", in triple backticks (e.g. ```code goes here```)"


TOOL_SYSTEM_PROMPT = (
    "You have access to the following tools:\n{tool_text}"
    "Use the following format if using a tool:\n"
    "```\n"
    "Action: tool name (one of [{tool_names}]).\n"
    "Action Input: the input to the tool{format_prompt}.\n"
    "```\n"
)


class Aligned(Agent):
    def __init__(self) -> None:
        self.type = "aligned"

    def _build_tool_text(self, functions: Dict[str, "Function"]) -> str:
        tool_text = ""
        for func in functions.values():
            param_text = ""
            for param in func.parameters:
                required = ", required" if param.name in func.required else ""
                enum = ", should be one of [{}]".format(", ".join(param.enum)) if len(param.enum) else ""
                item_type = ", where each item should be {}".format(param.item_type) if param.item_type else ""
                param_text += "  - {name} ({type}{required}): {desc}{enum}{item_type}\n".format(
                    name=param.name,
                    type=param.type,
                    required=required,
                    desc=param.description,
                    enum=enum,
                    item_type=item_type,
                )

            tool_text += "> Tool Name: {name}\nTool Description: {desc}\nTool Args:\n{args}\n".format(
                name=func.name, desc=func.description, args=param_text
            )

        return tool_text

    def build_prompt(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        functions = get_functions(tools)
        tool_prompt = TOOL_SYSTEM_PROMPT.format(
            tool_text=self._build_tool_text(functions),
            tool_names=", ".join(functions.keys()),
            format_prompt=CODE_FORMAT_PROMPT if "code_interpreter" in functions else JSON_FORMAT_PROMPT,
        )
        agent_messages = []
        if messages[0]["role"] == "system":
            system_message, messages = messages[0], messages[1:]
            agent_messages.append({"role": "system", "content": system_message["content"] + tool_prompt})
        else:
            agent_messages.append({"role": "system", "content": tool_prompt})

        for message in messages:
            if message["role"] == "function":
                tool_call: Dict[str, str] = json.loads(message["content"])
                function_repr = "Action: {}\nAction Input: {}\n".format(
                    tool_call.get("name", ""), tool_call.get("arguments", "")
                )
                agent_messages.append({"role": "assistant", "content": function_repr})
            elif message["role"] == "tool":  # no observation role
                agent_messages.append({"role": "user", "content": message["content"]})
            else:
                agent_messages.append(message)

        return agent_messages

    def extract_tool(self, answer: str, tools: Dict[str, Any]) -> Union[str, Tuple[str, str]]:
        functions = get_functions(tools)
        regex = re.compile(r"Action:\s*([a-zA-Z0-9_]+).*?Action Input:\s*(.*)", re.DOTALL)
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
            return answer

    def get_stop_word(self) -> None:
        return None


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
                        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    aligned = Aligned()
    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    print(aligned.build_prompt(messages, tools))

    answer = """Action: get_current_weather\nAction Input: {\n"location": "Boston, MA"\n}"""
    print(aligned.extract_tool(answer, tools))

    answer = "The weather in Boston is currently sunny."
    print(aligned.extract_tool(answer, tools))
