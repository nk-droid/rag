class Runtime:
    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def add_step(self, name: str):
        raise NotImplementedError

    def run_step(self, name: str, func, *args, **kwargs):
        return func(*args, **kwargs)

    def log(self, message: str):
        raise NotImplementedError
