from typing import Any

from pipeline.registry import REGISTRY
from infra.logging.runtime.factory import get_runtime

class RAGOrchestrator:
    def __init__(self, config):
        self.config = config
        self.runtime = get_runtime(config)
        self.setup_steps = config.get("init_pipeline", {}).get("steps", [])
        self.steps = config["pipeline"]["steps"]
        self._started = False

    def _ensure_started(self, msg: str):
        if not self._started:
            self.runtime.start(msg)
            self._started = True

    def _expand_step(self, step: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
        name = step["name"]
        raw_component = step["component"]
        step_options = {k: v for k, v in step.items() if k not in {"name", "component", "options"}}
        per_component_options = step.get("options", {})

        if isinstance(raw_component, str):
            return [(name, raw_component, step_options)]

        if not isinstance(raw_component, list):
            raise TypeError(
                f"Step '{name}' has invalid component type {type(raw_component)!r}. "
                "Expected str or list[str]."
            )

        expanded: list[tuple[str, str, dict[str, Any]]] = []
        for component_name in raw_component:
            if not isinstance(component_name, str):
                raise TypeError(
                    f"Step '{name}' has invalid list component type {type(component_name)!r}. "
                    "Expected list[str]."
                )

            component_step_options = dict(step_options)
            if isinstance(per_component_options, dict):
                scoped = per_component_options.get(component_name, {})
                if isinstance(scoped, dict):
                    component_step_options.update(scoped)

            expanded.append((f"{name}:{component_name}", component_name, component_step_options))

        return expanded

    def _execute_steps(self, steps, state):
        for step in steps:
            expanded_steps = self._expand_step(step)
            for run_name, component_name, step_options in expanded_steps:
                component = REGISTRY[component_name]

                step_state = dict(state)
                step_state["_step"] = {
                    "name": run_name,
                    "component": component_name,
                    **step_options,
                }

                state = self.runtime.run_step(
                    run_name,
                    component,
                    step_state,
                    self.config
                )

                state.pop("_step", None)

        return state

    def initialize(self, state):
        self._ensure_started("Initializing RAG System")

        for step in self.setup_steps:
            for run_name, _, _ in self._expand_step(step):
                self.runtime.add_step(run_name)

        if self.setup_steps:
            state = self._execute_steps(self.setup_steps, state)

        if self._started:
            self.runtime.stop("System Ready")
            self._started = False

        return state

    def run(self, state):
        self._ensure_started("Inferencing the System")
        for step in self.steps:
            for run_name, _, _ in self._expand_step(step):
                self.runtime.add_step(run_name)
        state = self._execute_steps(self.steps, state)

        if self._started:
            self.runtime.stop("")
            self._started = False

        return state
