# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP / REST / A2A Agent Endpoint Adapter.

Interfaces any remote deployed agent system accessible via HTTP/REST/A2A endpoints
(e.g., Cloud Run, Vertex AI Agent Engine, FastAPI, OpenAI-compatible servers).
"""

import json
import time
from typing import Any, Dict, List, Optional
import requests
from agent_eval_framework.adapters.base import BaseAgentAdapter, normalize_adapter_output
from agent_eval_framework.utils.logger import get_logger

log = get_logger(__name__)


class HttpAgentAdapter(BaseAgentAdapter):
    """Adapter for testing remote deployed agents over HTTP/REST."""

    def __init__(
        self,
        endpoint_url: str,
        headers: Optional[Dict[str, str]] = None,
        method: str = "POST",
        prompt_payload_key: str = "prompt",
        response_json_path: str = "response",
        trajectory_json_path: Optional[str] = "trajectory",
        timeout: int = 60,
        **kwargs,
    ):
        """Initializes the HTTP Agent Adapter.

        Args:
            endpoint_url: Full URL of the remote agent API.
            headers: HTTP headers (e.g. Authorization tokens).
            method: HTTP method ("POST" or "GET").
            prompt_payload_key: Key in the JSON request payload for the user prompt.
            response_json_path: Dot-separated path to extract response text from JSON (e.g. "choices.0.message.content").
            trajectory_json_path: Dot-separated path to extract tool trajectory list.
            timeout: Request timeout in seconds.
        """
        self.endpoint_url = endpoint_url
        self.headers = headers or {"Content-Type": "application/json"}
        self.method = method.upper()
        self.prompt_payload_key = prompt_payload_key
        self.response_json_path = response_json_path
        self.trajectory_json_path = trajectory_json_path
        self.timeout = timeout
        super().__init__(**kwargs)

    def load_agent(self, **kwargs) -> Any:
        return self.endpoint_url

    def _extract_path(self, data: Any, path: str) -> Any:
        """Extracts nested value from JSON object using dot path."""
        if not path:
            return data
        curr = data
        for part in path.split("."):
            if isinstance(curr, dict):
                curr = curr.get(part)
            elif isinstance(curr, list) and part.isdigit():
                idx = int(part)
                curr = curr[idx] if idx < len(curr) else None
            else:
                return None
        return curr

    def get_response(self, prompt: str) -> Dict[str, Any]:
        """Sends HTTP request to agent endpoint and extracts response."""
        start_time = time.time()
        try:
            payload = {self.prompt_payload_key: prompt}
            if self.method == "POST":
                resp = requests.post(
                    self.endpoint_url,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout,
                )
            else:
                resp = requests.get(
                    self.endpoint_url,
                    params=payload,
                    headers=self.headers,
                    timeout=self.timeout,
                )

            latency = round(time.time() - start_time, 3)
            resp.raise_for_status()
            data = resp.json()

            # Extract response and trajectory
            response_text = self._extract_path(data, self.response_json_path) or str(data)
            trajectory = self._extract_path(data, self.trajectory_json_path) if self.trajectory_json_path else []
            if not isinstance(trajectory, list):
                trajectory = []

            events = data.get("multi_agent_events", [])
            metadata = {
                "status_code": resp.status_code,
                "latency_seconds": latency,
                "endpoint": self.endpoint_url,
            }

            return normalize_adapter_output(
                response_text=str(response_text),
                trajectory=trajectory,
                multi_agent_events=events,
                metadata=metadata,
            )

        except Exception as e:
            log.error(f"HTTP Agent request failed: {e}", exc_info=True)
            return normalize_adapter_output(
                response_text=f"HTTP Error: {e}",
                metadata={"latency_seconds": round(time.time() - start_time, 3)},
                error=str(e),
            )
