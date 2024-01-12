import os
from ..function_prompt.prompt import REACT_PROMPT, json_format_prompt, json_format_warning_prompt
from ..function_prompt.tools_desc import get_react_tool_descriptions
from typing import List


class ReAct(object):

    def __init__(self, query: str, tools_name: List[str] = ['code_interpreter']):
        self.query = query
        self.react_template = REACT_PROMPT
        self.prompt = ''
        # self.tools_name = ['river_environment', 'code_interpreter]
        # self.tools_name = ['code_interpreter']
        self.tools_name = ['river_environment']

    def build_prompt(self):
        query = self.query
        tools_text = self._build_tools_text()
        tools_name_text = self._build_tools_name_text()
        planning_prompt = self.react_template.format(
            query=query,
            tools_text=tools_text,
            tools_name_text=tools_name_text,
            json_format_prompt=json_format_prompt if 'code_interpreter' not in self.tools_name else '',
            json_format_warning_prompt=json_format_warning_prompt if 'code_interpreter' not in self.tools_name else ''
        )

        self.prompt = planning_prompt
        return planning_prompt

    def _build_tools_text(self):
        return "\n".join(get_react_tool_descriptions(self.tools_name))

    def _build_tools_name_text(self):
        return "\n".join(self.tools_name)

    def build_observation(self, observation):
        return f'\nObservation: {observation}\nThought:'

    def get_stop_words_list(self):
        return ['Observation:', 'Observation:\n', 'Observ']
