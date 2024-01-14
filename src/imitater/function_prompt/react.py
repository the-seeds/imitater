import os
from ..function_prompt.prompt import REACT_PROMPT, json_format_prompt, json_format_warning_prompt
from typing import Dict, Any


class ReAct(object):

    def __init__(self, query: str, tools: Dict[str, Any]):
        self.query = query
        self.react_template = REACT_PROMPT
        self.prompt = ''
        self.tools_map = self.filter_tools_map(tools)

    def filter_tools_map(self, tools):
        tools_map = {tool['function']['name']: tool['function']
                     for tool in tools}
        if 'code_interpreter' in tools_map.keys():
            tools_map = {'code_interpreter': tools_map['code_interpreter']}
        return tools_map

    def build_prompt(self):
        query = self.query
        tools_text = self._build_tools_text()
        tools_name_text = self._build_tools_name_text()
        planning_prompt = self.react_template.format(
            query=query,
            tools_text=tools_text,
            tools_name_text=tools_name_text,
            json_format_prompt=json_format_prompt if 'code_interpreter' not in self.tools_map.keys() else '',
            json_format_warning_prompt=json_format_warning_prompt if 'code_interpreter' not in self.tools_map.keys() else ''
        )
        self.prompt = planning_prompt
        return planning_prompt

    def _build_tools_text(self):
        tool_descs = []
        for tool in self.tools_map.values():
            tool_desc = (
                f"> Tool Name: {tool['name']}\n"
                f"Tool Description: {tool['description']}\n"
                f"Tool Args: {str(tool['parameters'])}\n"
            )
            tool_descs.append(tool_desc)
        return "\n".join(tool_descs)

    def _build_tools_name_text(self):
        return "\n".join(self.tools_map.keys())

    def build_observation(self, observation):
        return f'\nObservation: {observation}\nThought:'

    def get_stop_words_list(self):
        return ['Observation:', 'Observation:\n', 'Observ']

    def get_stop_word_id(self):
        return 37763
