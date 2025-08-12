from typing import Any, List, Optional, Union

from pydantic import BaseModel

class Message(BaseModel):
    """Message sent to the Model"""

    # The role of the message author.
    # One of system, user, assistant, or tool.
    role: str

    # The contents of the message.
    content: Optional[Union[List[Any], str]] = None

    # The reasoning content of the message.
    reasoning_content: Optional[Union[List[Any], str]] = None

    # 该信息是否为assistant对 user 的追问信息
    is_follow_up: Optional[bool] = None