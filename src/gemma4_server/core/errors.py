class ValidationError(ValueError):
    pass

class QueueFullError(RuntimeError):
    pass

class StoreFullError(RuntimeError):
    pass

class WorkerNotReadyError(RuntimeError):
    def __init__(self, message="TPU worker is not ready", health=None):
        super().__init__(message)
        self.health = health or {}
