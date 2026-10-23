"""Real Laya adapter. Imports and weight loading are deferred until requested."""
import math
import os
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import RLock
from time import perf_counter


def package_versions():
    """Collect versions of the prototype's inference and UI dependencies.

    Returns:
        A mapping from package names to installed versions, with "not installed"
        for packages whose distribution metadata is unavailable.
    """
    result = {}
    for name in ("laya", "torch", "transformers", "streamlit"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not installed"
    return result


class LayaBackend:
    def __init__(self):
        """Initialize an unloaded backend with a lock and empty load metadata."""
        self.router = None
        self.load_ms = None
        self.load_warnings = []
        self.revision = None
        self.lock = RLock()

    def load(self):
        """Load and cache the English Laya checkpoint on CPU, once per adapter.

        Set project-local Hugging Face and Torch cache locations unless already
        configured through the environment. First use may download weights unless
        offline mode is enabled. Record successful load time, revision, and warnings.
        Loading failures propagate and leave the backend available for a later retry.

        Raises:
            ImportError: Required inference dependencies cannot be imported.
            OSError: Checkpoint files cannot be retrieved or read.
            ValueError: The checkpoint or its configuration is incompatible.
        """
        with self.lock:
            if self.router is not None:
                return
            # Keep download/cache writes inside this project, not the user's global cache.
            cache = Path(__file__).resolve().parent / ".cache"
            os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
            os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
            start = perf_counter()
            from laya import Router
            router = Router(device="cpu", max_loaded=1)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                agent = router.load("english")
            self.load_warnings = list(dict.fromkeys(str(item.message) for item in caught))
            self.revision = getattr(agent, "revision", None)
            self.router = router
            self.load_ms = (perf_counter() - start) * 1000

    def predict(self, state, questions):
        """Evaluate typed questions using the cached English checkpoint.

        Load the model on first use and serialize inference calls with the adapter's
        lock. Loading and inference failures propagate to the caller.

        Args:
            state: Text or structured input state to classify.
            questions: Mapping of question identifiers to Laya question definitions.

        Returns:
            The raw Laya result, including answers, probabilities, usage, and routing.
        """
        self.load()
        with self.lock:
            return self.router.predict(state, questions, model="english")

    def metadata(self):
        """Describe the configured model and any successful load.

        Returns:
            A dictionary containing model/checkpoint identifiers, CPU device, load
            time, revision, captured warnings, and installed package versions. Load
            time and revision are None until a successful load has populated them.
        """
        return {"model": "convaiinnovations/laya", "checkpoint": "english",
                "device": "cpu", "load_ms": self.load_ms, "revision": self.revision,
                "load_warnings": self.load_warnings,
                "packages": package_versions()}


def read_choice(raw, question_id, criteria):
    """Validate and extract a choice and its answer confidence.

    Args:
        raw: Raw Laya result containing an answers mapping.
        question_id: Identifier of the answer to inspect.
        criteria: Allowed choice labels, typically the question's criteria map.

    Returns:
        A tuple containing the selected label and its floating-point confidence.

    Raises:
        KeyError: A required answer, question, or choice field is missing.
        TypeError: The response shape or choice cannot be used as expected.
        ValueError: The choice is not allowed, or confidence is missing, boolean,
            nonnumeric, non-finite, or outside the inclusive range [0, 1].
    """
    answer = raw["answers"][question_id]
    choice = answer["choice"]
    score = answer.get("answer_confidence")
    if choice not in criteria:
        raise ValueError(f"Unexpected choice for {question_id}: {choice!r}")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"Missing numeric answer_confidence for {question_id}")
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"Invalid answer_confidence for {question_id}")
    return choice, float(score)
