"""Evaluation kit for conversational mask editors.

clinician  the abstract simulated-clinician interface
runner     runs one dialogue, or a study, against any editor with begin_session / begin_image / step / end_image
outcomes   reach, gain, gain under a common stopping rule, improved and worse shares, simulated time, and the summary
table
bootstrap  clustered bootstrap by clinician and image
perturb    word errors and click jitter, for robustness runs
toy        a small example simulator; an illustration only, not an evaluation instrument
"""

from .bootstrap import cluster_bootstrap
from .clinician import Case, Observation, SimulatedClinician
from .outcomes import TimeCosts, gain, gain_stop, reach, simulated_time, summarize, summary_markdown
from .perturb import inject_word_errors, jitter_click
from .runner import DialogueRecord, ParleyEvalAdapter, run_dialogue, run_study

__all__ = [
    "Case",
    "Observation",
    "SimulatedClinician",
    "DialogueRecord",
    "ParleyEvalAdapter",
    "run_dialogue",
    "run_study",
    "TimeCosts",
    "reach",
    "gain",
    "gain_stop",
    "simulated_time",
    "summarize",
    "summary_markdown",
    "cluster_bootstrap",
    "inject_word_errors",
    "jitter_click",
]
