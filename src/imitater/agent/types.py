from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union


class Agent(ABC):
    type: str

    @abstractmethod
    def build_prompt(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        r"""
        Formats messages for agent inference.
        """
        ...

    @abstractmethod
    def extract_tool(self, answer: str, tools: List[Dict[str, Any]]) -> Union[str, Tuple[str, str]]:
        r"""
        Extracts tool name and arguments from model outputs.
        """
        ...

    @abstractmethod
    def get_stop_word(self) -> Optional[str]:
        r"""
        Gets the stop word.
        """
        ...
