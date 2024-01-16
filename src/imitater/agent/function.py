from dataclasses import dataclass
from typing import Any, Dict, List, Union

from typing_extensions import Literal


_PARAM_TYPE = {"string": str, "integer": int, "float": (int, float), "bool": bool}


@dataclass
class Parameter:
    name: str
    type: Literal["string", "integer", "float", "bool"]
    description: str


@dataclass
class Function:
    name: str
    description: str
    parameters: List[Parameter]
    required: List[str]

    def get_param_dict(self) -> Dict[str, Dict[str, str]]:
        param_dict = {}
        for param in self.parameters:
            param_dict[param.name] = {"type": param.type, "description": param.description}

        return param_dict

    def verify_param(self, name: str, value: Any) -> bool:
        param_dict = self.get_param_dict()
        if name not in param_dict:
            return False

        if not isinstance(value, _PARAM_TYPE[param_dict[name]["type"]]):
            return False

        return True


def get_functions(tools: List[Dict[str, Any]]) -> Dict[str, Function]:
    r"""
    Converts tools to functions.
    """
    functions = {}
    for tool in tools:
        if tool["type"] == "code_interpreter":
            functions["code_interpreter"] = Function(
                name="code_interpreter",
                description=(
                    "A Python code interpreter that executes Python code. "
                    "Enclose the code within triple backticks, e.g. ```code goes here```."
                ),
                parameters=[Parameter(name="code", type="string", description="Python code to execute.")],
                required=["code"],
            )
            break

        elif tool["type"] == "function":
            function_data: Dict[str, Union[str, Dict[str, Any]]] = tool["function"]
            properties: Dict[str, Dict[str, Any]] = function_data["parameters"].get("properties", {})
            parameters = [
                Parameter(name=name, type=prop.get("type", ""), description=prop.get("description", ""))
                for name, prop in properties.items()
            ]
            functions[function_data["name"]] = Function(
                name=function_data["name"],
                description=function_data["description"],
                parameters=parameters,
                required=function_data["parameters"].get("required", []),
            )

    return functions


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
    functions = get_functions(tools)
    print(functions)
    print(functions["get_current_weather"].get_param_dict())
    print(functions["get_current_weather"].verify_param("city", "San Francisco"))
    print(functions["get_current_weather"].verify_param("location", 0))
    print(functions["get_current_weather"].verify_param("location", "San Francisco"))
