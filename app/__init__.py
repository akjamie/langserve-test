from langserve import CustomUserType


class QaRequest(CustomUserType):
    """Input for the chat endpoint."""
    user_input: str


class QaResponse(CustomUserType):
    """Input for the chat endpoint."""
    user_input: str
    result: str
