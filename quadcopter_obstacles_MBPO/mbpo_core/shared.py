import torch

from .sampling import SampleBuffer


class SafetySampleBuffer(SampleBuffer):
    """Replay buffer that stores an extra boolean violation flag per transition."""

    COMPONENT_NAMES = (*SampleBuffer.COMPONENT_NAMES, "violations")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_buffer("violations", torch.bool, [])
