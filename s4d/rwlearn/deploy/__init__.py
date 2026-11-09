"""Science-only deployment API (docs/realworld_deployment.md). The client-side names (contract, frames, history,
preprocessing, refusals) never import torch; ``DeployPolicy`` is a lazy attribute. No transport or robot I/O."""
from s4d.rwlearn.deploy.contracts import InputSpec
from s4d.rwlearn.deploy.errors import (
    BindingRefused,
    ContractViolation,
    DeployRefusal,
    IncompleteHistory,
    IrregularHistory,
    SessionMismatch,
    Unsynchronized,
)
from s4d.rwlearn.deploy.history import Frame, HistoryWindow, ObservationWindow
from s4d.rwlearn.deploy.preprocessing import preprocess_frame

__all__ = ["BindingRefused", "ContractViolation", "DeployPolicy", "DeployRefusal", "Frame", "HistoryWindow",
           "IncompleteHistory", "InputSpec", "IrregularHistory", "ObservationWindow", "SessionMismatch",
           "Unsynchronized", "preprocess_frame"]


def __getattr__(name):
    if name == "DeployPolicy":
        from s4d.rwlearn.deploy.policy import DeployPolicy

        return DeployPolicy
    raise AttributeError(name)
