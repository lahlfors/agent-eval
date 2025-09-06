from abc import ABC, abstractmethod
from typing import Any, Callable
# import requests  # Note: In a real scenario, this would be a dependency.

class SystemAdapter(ABC):
    """
    An abstract base class representing an interface to a system under test.
    This allows the evaluation pipeline to be generic and work with any system
    that can be "adapted" to this interface.
    """
    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """
        Executes the system with the given input and returns the output.
        """
        pass


class LibraryAdapter(SystemAdapter):
    """
    An adapter for a system that is a Python callable (e.g., a function or a
    class method).
    """
    def __init__(self, tool_function: Callable[[Any], Any]):
        self._tool_function = tool_function

    def execute(self, input_data: Any) -> Any:
        """
        Executes the system by calling the provided function.
        """
        try:
            return self._tool_function(input_data)
        except Exception as e:
            # In a real-world scenario, you might want more robust error handling.
            print(f"Error executing library tool: {e}")
            return None


class ApiAdapter(SystemAdapter):
    """
    An adapter for a system that is exposed via an HTTP API endpoint.
    """
    def __init__(self, api_url: str, method: str = "POST", headers: dict = None):
        self.api_url = api_url
        self.method = method.upper()
        self.headers = headers or {"Content-Type": "application/json"}

    def execute(self, input_data: Any) -> Any:
        """
        Executes the system by making an HTTP request to the API endpoint.

        Note: This is a placeholder implementation. To make this functional,
        you would need to install a library like 'requests' or 'httpx' and
        uncomment the import at the top of the file.
        """
        print(f"--- MOCK API CALL to {self.method} {self.api_url} ---")
        print(f"--- Headers: {self.headers}")
        print(f"--- Body/Params: {input_data}")
        print("--- Returning mock response: {'status': 'ok'} ---")
        # In a real implementation, you would use a library like requests:
        #
        # try:
        #     response = requests.post(self.api_url, json=input_data, headers=self.headers)
        #     response.raise_for_status()
        #     return response.json()
        # except requests.RequestException as e:
        #     print(f"Error calling API endpoint {self.api_url}: {e}")
        #     return None
        return {"status": "ok"} # Return a mock response for demonstration
