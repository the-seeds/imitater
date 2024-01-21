from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Union

from typing_extensions import Literal


_PARAM_TYPE = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "float": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass
class ParameterValue:
    type: Literal["string", "number", "integer", "float", "boolean", "array", "object"]
    description: str
    enum: List[str]
    item_type: str


@dataclass
class Parameter(ParameterValue):
    name: str


@dataclass
class Function:
    name: str
    description: str
    parameters: List[Parameter]
    required: List[str]

    def get_param_mapping(self) -> Dict[str, "ParameterValue"]:
        param_mapping = {}
        for param in self.parameters:
            param_dict = asdict(param)
            name = param_dict.pop("name")
            param_mapping[name] = ParameterValue(**param_dict)

        return param_mapping

    def verify_param(self, name: str, value: Any) -> bool:
        param_mapping = self.get_param_mapping()
        if name not in param_mapping:
            return False

        if not isinstance(value, _PARAM_TYPE[param_mapping[name].type]):
            return False

        if len(param_mapping[name].enum) and value not in param_mapping[name].enum:
            return False

        if param_mapping[name].item_type and not isinstance(value[0], _PARAM_TYPE[param_mapping[name].item_type]):
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
                Parameter(
                    name=name,
                    type=prop.get("type", ""),
                    description=prop.get("description", ""),
                    enum=prop.get("enum", []),
                    item_type=prop.get("items", {}).get("type", ""),
                )
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
                        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    functions = get_functions(tools)
    print(functions)
    print(functions["get_current_weather"].get_param_mapping())
    print(functions["get_current_weather"].verify_param("city", "San Francisco"))
    print(functions["get_current_weather"].verify_param("location", 0))
    print(functions["get_current_weather"].verify_param("location", "San Francisco"))
    print(functions["get_current_weather"].verify_param("unit", "test"))
    print(functions["get_current_weather"].verify_param("unit", "fahrenheit"))
