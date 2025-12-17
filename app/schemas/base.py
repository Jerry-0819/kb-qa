from pydantic import BaseModel
from typing import Literal

class ChatPayload(BaseModel):
    query: str
    k: int = 3
    mode: Literal["chain", "agent"] = "chain"
