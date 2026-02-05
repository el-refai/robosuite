import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from robosuite.environments.custom.base_env import BaseEnv


class ApiBase(ABC):
    """Base class for tool APIs.

    Guidelines:
    - Expose public functions via `functions()`.
    - Each function should have a Google-style docstring with Args/Returns.
    - `combined_doc()` returns a standardized aggregate doc for prompts.
    - If your API needs access to the environment, implement `set_env(env)`.

    Example:
    class GraspPlanApi(ApiBase):
        def functions(self) -> dict[str, Callable[..., Any]]:
            # optionally, we can expose only one of the functions if needed
            return {"grasp_plan": self.grasp_plan, "grasp_plan_sim": self.grasp_plan_sim}

        def grasp_plan(self, depth: np.ndarray, intrinsics: np.ndarray) -> list[dict]:
            \"""
            Plan parallel-jaw grasps from a depth map.

            Args:
            depth: (H, W) float32 meters.
            intrinsics: (3, 3) float32 pinhole intrinsics.

            Returns:
            List of dicts: {pose: (4,4) float32, width: float, score: float}.
            \"""
            ...

        def grasp_plan_sim(self, depth: np.ndarray, intrinsics: np.ndarray) -> list[dict]:
            ...
    """

    def __init__(self, env: BaseEnv) -> None:
        self._env = env

    @abstractmethod
    def functions(self) -> dict[str, Callable[..., Any]]:
        """Return mapping of public function name -> callable."""

    def combined_doc(self) -> str:
        """Aggregate function docs in a simple, consistent format.

        Format per function:
            name(signature)
              Summary: first line of function doc
              Doc: full function docstring (Google style recommended)
        """
        # we need to discuss this design further down the line
        lines: list[str] = []
        # lines: list[str] = [f"API: {self.__class__.__name__}", ""]
        for name, fn in self.functions().items():
            try:
                sig = str(inspect.signature(fn))
            except Exception:
                sig = "(…)"
            doc = inspect.getdoc(fn) or ""
            # first = doc.splitlines()[0] if doc else ""
            lines.append(f"{name}{sig}")
            # if first:
            #     lines.append(f"  Summary: {first}")
            if doc:
                lines.append("  Doc:")
                lines.extend(f"    {ln}" for ln in doc.splitlines())
            lines.append("")
        return "\n".join(lines).strip()


_API_FACTORIES: dict[str, Callable[[], ApiBase]] = {}


def register_api(name: str, factory: Callable[[], ApiBase]) -> None:
    _API_FACTORIES[name] = factory


@lru_cache(maxsize=256)
def get_api(name: str) -> Callable[[BaseEnv], ApiBase]:
    if name not in _API_FACTORIES:
        raise KeyError(f"API '{name}' not registered")
    return _API_FACTORIES[name]


def list_apis() -> list[str]:
    return list(_API_FACTORIES.keys())
