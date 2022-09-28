class EtcWorkflowException(Exception):
    """A base class for EtcWorkflow exceptions."""

class NoAsterException(EtcWorkflowException):
    """Raised when there is no Aster products available"""

class NoSentinelException(EtcWorkflowException):
    """Raised when there is no Sentinel-2 products available"""

class RuntimeMinioException(EtcWorkflowException):
    """Raised when there is no Sentinel-2 products available"""