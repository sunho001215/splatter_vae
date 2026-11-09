"""Typed deployment refusals (torch-free). Every refusal is a ``ValueError`` subclass, so callers that catch
``ValueError`` keep working; the networking layer maps the class (never the message text) to a wire error code."""


class DeployRefusal(ValueError):
    """Base class of every refusal raised by ``s4d.rwlearn.deploy``."""


class ContractViolation(DeployRefusal):
    """An input violates the policy's contract (missing/extra session choice, malformed frame, calibration, depth
    provenance, pose, noise)."""


class SessionMismatch(DeployRefusal):
    """The window/history belongs to no live session of this policy (or a reset reused a session id)."""


class IncompleteHistory(DeployRefusal):
    """Fewer than the trained number of frames (steps) of the current run, a missing frame, or missing M5 evidence."""


class IrregularHistory(DeployRefusal):
    """Frames are not at the trained spacing (seq gap inside a history, irregular timestamps)."""


class Unsynchronized(DeployRefusal):
    """Multi-view frames are not a synchronised set (mutual nearest within half a period, M5 robot motion)."""


class BindingRefused(DeployRefusal):
    """The measured pose / gripper register / ids are not bound to the observation as in training."""
