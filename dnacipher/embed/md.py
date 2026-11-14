""" Imports all the currently supported models.
"""

from .enformer_embed import EnformerEmbed
from .borzoi_embed import BorzoiEmbed
from .borzoi_2048_embed import BorzoiShortEmbed
from .borzoi_8192_embed import Borzoi8kEmbed
from .borzoi_16384_embed import Borzoi16kEmbed

model_classes = [EnformerEmbed, BorzoiEmbed, BorzoiShortEmbed, Borzoi8kEmbed, Borzoi16kEmbed]

model_names = [class_.model_name for class_ in model_classes]
