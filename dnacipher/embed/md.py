""" Imports all the currently supported models.
"""

from .enformer_embed import EnformerEmbed
from .borzoi_embed import BorzoiEmbed

model_classes = [EnformerEmbed, BorzoiEmbed]

model_names = [class_.model_name for class_ in model_classes]
