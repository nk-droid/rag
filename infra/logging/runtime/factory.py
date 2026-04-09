from infra.logging.runtime.rich_runtime import RichRuntime
from infra.logging.runtime.simple_runtime import SimpleRuntime
from infra.logging.runtime.silent_runtime import SilentRuntime

def get_runtime(config):
    mode = config.get("runtime", {}).get("mode", "cli")

    if mode == "cli":
        return RichRuntime()
    elif mode == "api":
        return SilentRuntime()

    return SimpleRuntime()