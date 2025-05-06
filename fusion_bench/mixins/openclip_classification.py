import logging

from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.models.open_clip import ImageClassifier, ImageEncoder

log = logging.getLogger(__name__)


class OpenCLIPClassificationMixin(LightningFabricMixin):
    _train_processor = None
    _test_processor = None
