from infra.logging.runtime.base import Runtime

class SilentRuntime(Runtime):
    def start(self, msg: str):
        pass

    def stop(self, msg: str = None):
        pass

    def add_step(self, name):
        pass

    def run_step(self, name, func, *args, **kwargs):
        return func(*args, **kwargs)
    
    def log(self, message: str):
        pass