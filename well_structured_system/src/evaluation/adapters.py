"""Provides adapters for interfacing with different system types.

This module defines a set of "Adapters," which are classes that provide a
consistent interface for the evaluation pipeline to interact with various kinds
of systems under test. For example, the system might be a simple Python function,
a class, or an external API endpoint. Each adapter handles the specific logic
for executing that type of system.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable

class SystemAdapter(ABC):
    """Abstract base class for a system adapter.

    This class defines the standard interface required for any system to be
    evaluated by the pipeline. It ensures that the pipeline can remain generic,
    simply calling the `execute` method regardless of whether the underlying
    system is a local function or a remote API.
    """
    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """Executes the system with the given input and returns the output.

        Args:
            input_data: The input data to be passed to the system. The format
                        is dependent on what the specific system expects.

        Returns:
            The output produced by the system.
        """
        pass

class LibraryAdapter(SystemAdapter):
    """An adapter for a system that is a Python callable.

    This adapter is used when the system under test is a Python function or a
    class method that can be directly imported and called.

    Attributes:
        _tool_function: The Python callable to be executed.
    """
    def __init__(self, tool_function: Callable[[Any], Any]):
        """Initializes the LibraryAdapter.

        Args:
            tool_function: The Python callable (e.g., a function or method)
                           that represents the system to be tested.
        """
        self._tool_function = tool_function

    def execute(self, input_data: Any) -> Any:
        """Executes the system by invoking the provided callable.

        Args:
            input_data: The input data to be passed to the callable.

        Returns:
            The return value of the callable, or None if an exception occurs.
        """
        try:
            return self._tool_function(input_data)
        except Exception as e:
            print(f"Error executing library tool: {e}")
            return None

class ApiAdapter(SystemAdapter):
    """An adapter for a system exposed via an HTTP API endpoint.

    This adapter handles the logic of making an HTTP request to an external
    system. It is configured with the URL, HTTP method, and headers for the
    request.

    Note:
        This is a mock implementation for demonstration purposes and does not
        make a real HTTP request. A production version would require a library
        like `requests` or `httpx`.

    Attributes:
        api_url: The URL of the API endpoint.
        method: The HTTP method to use (e.g., "POST", "GET").
        headers: A dictionary of headers to include in the request.
    """
    def __init__(self, api_url: str, method: str = "POST", headers: dict = None):
        """Initializes the ApiAdapter.

        Args:
            api_url: The URL of the API endpoint to call.
            method: The HTTP method to use for the request. Defaults to "POST".
            headers: A dictionary of HTTP headers. Defaults to a standard
                     JSON content-type header.
        """
        self.api_url = api_url
        self.method = method.upper()
        self.headers = headers or {"Content-Type": "application/json"}

    def execute(self, input_data: Any) -> Any:
        """Executes the system by making a mock HTTP request.

        This implementation prints the details of the would-be API call and
        returns a predefined mock response.

        Args:
            input_data: The data to be sent as the body or parameters of the
                        request.

        Returns:
            A mock dictionary response: `{"status": "ok"}`.
        """
        print(f"--- MOCK API CALL to {self.method} {self.api_url} ---")
        print(f"--- Headers: {self.headers}")
        print(f"--- Body/Params: {input_data}")
        print("--- Returning mock response: {'status': 'ok'} ---")
        return {"status": "ok"}
