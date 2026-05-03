from infra.logging.runtime.base import Runtime
import time

class SimpleRuntime(Runtime):
    def start(self, msg: str):
        print(msg)

    def stop(self, msg: str = None):
        print(msg)

    def add_step(self, name):
        pass

    def run_step(self, name, func, *args, **kwargs):
        print(f"Running: {name}")
        start = time.time()

        result = func(*args, **kwargs)

        duration = round(time.time() - start, 2)
        print(f"✔ {name} ({duration}s)")

        return result