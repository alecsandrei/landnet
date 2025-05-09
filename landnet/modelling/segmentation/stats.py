from __future__ import annotations

from torchmetrics import MetricCollection
from torchmetrics.segmentation import DiceScore, GeneralizedDiceScore, MeanIoU


class SegmentationMetricCollection(MetricCollection):
    def __init__(self, num_classes: int, prefix: str | None = None):
        self._num_classes = num_classes
        super().__init__(
            {
                'dice_score': DiceScore(self._num_classes),
                'generalized_dice_score': GeneralizedDiceScore(
                    self._num_classes
                ),
                'mIoU': MeanIoU(self._num_classes),
            },
            prefix=prefix,
        )
