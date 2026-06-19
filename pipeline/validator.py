from typing import Any

from pipeline.contracts import CONTRACTS
from pipeline.registry import REGISTRY

# Keys the orchestrator seeds into state before any step runs.
SEED_KEYS = frozenset({"query", "sources"})

def _step_components(step: dict[str, Any]) -> list[str]:
    component = step.get("component")
    if isinstance(component, list):
        return [str(c) for c in component]
    if component is None:
        return []
    return [str(component)]

def _validate_sequence(
    steps: list[dict[str, Any]],
    phase_label: str,
    seed: frozenset[str],
) -> list[str]:
    errors: list[str] = []
    available: set[str] = set(seed)

    for position, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"{phase_label}.steps[{position}] is not a mapping.")
            continue

        name = step.get("name", f"#{position}")
        components = _step_components(step)
        if not components:
            errors.append(f"{phase_label} step '{name}': missing 'component'.")
            continue

        for component in components:
            if component not in REGISTRY:
                errors.append(
                    f"{phase_label} step '{name}': unknown component '{component}' "
                    "(not in REGISTRY)."
                )
                continue

            contract = CONTRACTS.get(component)
            if contract is None:
                errors.append(
                    f"{phase_label} step '{name}': component '{component}' has no "
                    "contract in pipeline.contracts — cannot validate."
                )
                continue

            for group in contract.requires:
                if not (group & available):
                    need = " | ".join(sorted(group))
                    errors.append(
                        f"{phase_label} step '{name}' ({component}) requires "
                        f"[{need}], but no prior step produces it."
                    )

            available |= contract.produces

    return errors

def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    init_cfg = config.get("init_pipeline", {})
    run_cfg = config.get("pipeline", {})
    init_steps = init_cfg.get("steps", []) if isinstance(init_cfg, dict) else []
    run_steps = run_cfg.get("steps", []) if isinstance(run_cfg, dict) else []

    if not run_steps:
        errors.append("pipeline.steps is empty — nothing to run.")

    errors.extend(_validate_sequence(init_steps, "init_pipeline", SEED_KEYS))
    errors.extend(_validate_sequence(run_steps, "pipeline", SEED_KEYS))

    return errors
