"""The simulated-clinician interface. Implement `next_utterance`, `click`, `accept` and `target`."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Case:
    """One image to correct. `image`: uint8 (H, W, 3). `start_mask`: bool (H, W). `reference`: bool (H, W), the ground
    truth a simulator may build its target from.
    """

    case_id: str
    dataset: str
    image: np.ndarray
    start_mask: np.ndarray
    reference: np.ndarray


@dataclass
class Observation:
    """What a simulator sees at a turn: the case, the mask on screen and the number of turns taken so far."""

    case: Case
    mask: np.ndarray
    turn: int


class SimulatedClinician(ABC):
    """One simulated clinician. The runner calls `begin_image` once per case and then, at each turn, `accept`,
    `next_utterance` and `click`.
    """

    clinician_id: str = "clinician"
    accept_phrase: str | None = "that looks good"

    def begin_image(self, case: Case) -> None:
        """Called once per case, before the first turn."""

    @abstractmethod
    def target(self, case: Case) -> np.ndarray:
        """The mask this clinician wants on this case, bool (H, W). Outcomes are measured against it."""

    @abstractmethod
    def accept(self, obs: Observation) -> bool:
        """True when the clinician would stop here and accept the mask."""

    @abstractmethod
    def next_utterance(self, obs: Observation) -> str | None:
        """The next thing said, as written text. None gives up on this case."""

    @abstractmethod
    def click(self, obs: Observation, utterance: str) -> tuple[float, float] | None:
        """One click (x, y) in mask pixels that goes with this utterance, or None."""
