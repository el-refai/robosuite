import time

import requests


def post_with_retries(
    url: str, payload: dict, timeout_seconds: float = 15.0, retry_interval: float = 0.5
):
    """
    Retry POST requests for up to `timeout_seconds` of wall clock time.

    Raises RuntimeError if the time limit is exceeded.
    """
    deadline = time.time() + timeout_seconds

    last_err = None
    while time.time() < deadline:
        try:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            time.sleep(retry_interval)

    raise RuntimeError(
        f"Request to {url} failed after {timeout_seconds:.2f}s of retries. Last error: {last_err}"
    )