from typing import Optional

class ModelError(Exception):
    """Exception raised when an internal error occurs."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    def __str__(self) -> str:
        return str(self.message)


class ModelProviderError(ModelError):

    """Exception raised when a model provider returns an error."""

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        model_name: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        super().__init__(message, status_code)
        self.model_name = model_name
        self.model_id = model_id