from __future__ import annotations

import json
import mimetypes
import os
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import requests


class ComfyUIError(RuntimeError):
    pass


class ComfyUIClient:
    def __init__(
        self,
        base_url: str,
        timeout_connect: int = 10,
        timeout_read: int = 300,
        poll_interval: float = 1.5,
        poll_timeout: int = 900,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_connect = timeout_connect
        self.timeout_read = timeout_read
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def check_server(self) -> Dict[str, Any]:
        response = requests.get(
            self._url("/system_stats"),
            timeout=(self.timeout_connect, self.timeout_read),
        )
        response.raise_for_status()
        return response.json()

    def upload_image(self, image_path: str, subfolder: str = "") -> Dict[str, Any]:
        if not os.path.isfile(image_path):
            raise ComfyUIError(f"Input image not found: {image_path}")

        filename = os.path.basename(image_path)
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        with open(image_path, "rb") as f:
            files = {"image": (filename, f, content_type)}
            data = {"overwrite": "true"}
            if subfolder:
                data["subfolder"] = subfolder

            response = requests.post(
                self._url("/upload/image"),
                files=files,
                data=data,
                timeout=(self.timeout_connect, self.timeout_read),
            )

        response.raise_for_status()
        payload = response.json()

        if "name" not in payload:
            raise ComfyUIError(f"Unexpected upload response: {payload}")

        return payload

    def queue_prompt(self, prompt: Dict[str, Any]) -> str:
        payload = {"prompt": prompt}
        response = requests.post(
            self._url("/prompt"),
            json=payload,
            timeout=(self.timeout_connect, self.timeout_read),
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise ComfyUIError(
                f"ComfyUI prompt validation failed: {json.dumps(data, ensure_ascii=False)}"
            )

        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"ComfyUI did not return prompt_id: {data}")

        return str(prompt_id)

    def get_history_item(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        response = requests.get(
            self._url(f"/history/{prompt_id}"),
            timeout=(self.timeout_connect, self.timeout_read),
        )
        response.raise_for_status()
        data = response.json()
        if not data:
            return None
        return data.get(prompt_id)

    def wait_for_completion(self, prompt_id: str) -> Dict[str, Any]:
        start = time.time()

        while True:
            if time.time() - start > self.poll_timeout:
                raise ComfyUIError(f"Timed out waiting for ComfyUI prompt {prompt_id}")

            item = self.get_history_item(prompt_id)
            if item is not None:
                status = item.get("status", {})
                status_str = status.get("status_str", "")

                if status_str == "error":
                    raise ComfyUIError(f"ComfyUI prompt failed: {item}")

                outputs = item.get("outputs")
                if outputs:
                    return item

            time.sleep(self.poll_interval)

    def build_view_url(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> str:
        query = urlencode(
            {
                "filename": filename,
                "subfolder": subfolder,
                "type": folder_type,
            }
        )
        return self._url(f"/view?{query}")

    def download_output(
        self,
        filename: str,
        destination_path: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> str:
        url = self.build_view_url(
            filename=filename,
            subfolder=subfolder,
            folder_type=folder_type,
        )

        response = requests.get(
            url,
            timeout=(self.timeout_connect, self.timeout_read),
            stream=True,
        )
        response.raise_for_status()

        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        with open(destination_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        return destination_path