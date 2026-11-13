""" Imports all the currently supported models.
"""

from .enformer_embed import EnformerEmbed
from .borzoi_embed import BorzoiEmbed
from .borzoi_2048_embed import BorzoiShortEmbed

model_classes = [EnformerEmbed, BorzoiEmbed, BorzoiShortEmbed]

model_names = [class_.model_name for class_ in model_classes]
