from infra.logging.runtime.base import Runtime
import time

class SimpleRuntime(Runtime):
    def start(self):
        print("Initializing RAG system...")

    def stop(self):
        print("System ready.")

    def add_step(self, name):
        print(f"→ {name}")

    def run_step(self, name, func, *args, **kwargs):
        print(f"Running: {name}")
        start = time.time()

        result = func(*args, **kwargs)

        duration = round(time.time() - start, 2)
        print(f"✔ {name} ({duration}s)")

        return result