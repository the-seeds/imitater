from typing import List


CODE_INTERPRETER_DESC = {
    "name": "code_interpreter",
    "description": "interpreter for code, which use for executing python code",
    "fn_schema": {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": "interpreter for code, which use for executing python code",
            "parameters": [{
                "type": "string",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "the code to be executed, which Enclose the code within triple backticks (```) at the beginning and end of the code."
                    }
                }
            }],
        }
    }
}

RIVER_ENVIRONMENT_DESC = {
    "name": "get_river_environment",
    "description": "get river environment",
    "fn_schema": {
        "type": "function",
        "function": {
            "name": "get_river_environment",
            "description": "get river environment",
            "parameters":[{
                "type": "string",
                "properties": {
                    "river": {
                        "type": "string",
                        "description": "the river name"
                    },
                    "location": {
                        "type": "string",
                        "description": "the location of the river"
                    }
                }
            }]
        }
    }
}

TOOLS_MAP = {
    "code_interpreter": CODE_INTERPRETER_DESC,
    "river_environment": RIVER_ENVIRONMENT_DESC,
}


def get_react_tool_descriptions(tools_name: List[str]):
    tools = []
    for tool in tools_name:
        if tool in TOOLS_MAP.keys():
            tools.append(TOOLS_MAP[tool])

    tool_descs = []
    for tool in tools:
        tool_desc = (
            f"> Tool Name: {tool['name']}\n"
            f"Tool Description: {tool['description']}\n"
            f"Tool Args: {str(tool['fn_schema'])}\n"
        )
        tool_descs.append(tool_desc)
    return tool_descs
