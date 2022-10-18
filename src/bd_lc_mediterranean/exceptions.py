class WorkflowExecutionException(Exception):
    """A base class for WorkflowExecution exceptions."""

class NoAsterException(WorkflowExecutionException):
    """Raised when there is no Aster products available"""

class NoSentinelException(WorkflowExecutionException):
    """Raised when there is no Sentinel-2 products available"""

class RuntimeMinioException(WorkflowExecutionException):
    """Raised when there is no Sentinel-2 products available"""