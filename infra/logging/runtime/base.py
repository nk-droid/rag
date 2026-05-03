from infra.logging.recent_logs import add_log

class Runtime:
    def start(self, msg: str):
        raise NotImplementedError

    def stop(self, msg: str = None):
        raise NotImplementedError

    def add_step(self, name: str):
        raise NotImplementedError

    def run_step(self, name: str, func, *args, **kwargs):
        return func(*args, **kwargs)

    def log(self, message: str):
        add_log(str(message))
