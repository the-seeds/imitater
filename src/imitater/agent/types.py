from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union


class Agent(ABC):
    type: str

    @abstractmethod
    def build_prompt(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        r"""
        Formats messages for agent inference.

        Args:
            messages: the current messages.
            tools: the tool specification in the OpenAI format.

        Returns:
            messages: the formatted messages with tool information.
        """
        ...

    @abstractmethod
    def extract_tool(self, answer: str, tools: List[Dict[str, Any]]) -> Union[str, Tuple[str, str]]:
        r"""
        Extracts tool name and arguments from model outputs.

        Args:
            answer: the text to extract the tool from it.
            tools: the tool specification in the OpenAI format.

        Returns:
            response | (name, arguments): response text or tool name with JSON arguments if tool exists.
        """
        ...

    @abstractmethod
    def get_stop_word(self) -> Optional[str]:
        r"""
        Gets the stop word.

        Returns:
            stop_word (optional): the stop word.
        """
        ...
