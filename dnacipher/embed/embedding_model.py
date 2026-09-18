"""
Parent class for an embedding model.
"""

import abc

class EmbeddingModel:

    """ Defines a model which generates embeddings from input sequence.
    """

    model_name = None

    def __init__(self, device, input_seq_len=None, embed_seq_len=None, embed_seq_resolution=None, embed_features=None,
                 *args, **kwargs):
        """ Instantiates the embedding model.

        Args:

            input_seq_len (int): The length of the input sequence, as a nucleotide string, the model takes as input.
                                       Enformer example, takes as input 196,608 bp of input sequence.

            embed_seq_len (int): The output length of the model embedding, this is assumed to correspond to the
                                        centre of the input sequence, such that (model_input_seq_len-model_embed_seq_len)/2
                                        corresponds to the additional sequence context on either side of the output embeddings.
                                        Enformer for example, outputs embeddings representing the centre 114,688 bp of the input sequence.

            embed_seq_resolution (int): The number of nucleotides respresented by a given set of embedding features,
                                        such that model_embed_seq_len / model_embed_seq_resolution corresponds to
                                        the number of embeddings outputted from the model.
                                        Enformer for example, outputs an embedding at a 128bp resolution, along the centre 114,688 bp of the
                                        196,608 bp input sequence.

            embed_features (int): The number of features per embedding. Enformer as an example outputs 3072 features per
                                        embedding. So the total embedding size would be expected to be a matrix of size:
                                        (model_embed_seq_len / model_embed_seq_resolution) x model_embed_features

            device (str): Device the model will run on. (e.g. 'cuda:0').

            *args: Any additional *args will be parsed to setup_model() for setting up the specifying sequence 2 function model.

            **kwargs: Any additional optional arguments will be parsed to setup_model() for setting up the specifying sequence 2 function model.

        """

        self.input_seq_len = input_seq_len
        self.embed_seq_len = embed_seq_len
        self.edge_context = (input_seq_len - embed_seq_len) // 2
        self.embed_seq_resolution = embed_seq_resolution
        self.embed_features = embed_features

        self.model = self.setup_model(device, *args, **kwargs)

    def __str__(self):
        str_ = f"Model name: {self.model_name}.\n"+\
               f"Input seq length: {self.input_seq_len}.\n" +\
               f"Embed seq length: {self.embed_seq_len}.\n" +\
               f"Edge context length: {self.edge_context}.\n" +\
               f"Embed seq resolution: {self.embed_seq_resolution}.\n"+\
               f"Embed features: {self.embed_features}.\n\n"
        return str_

    @abc.abstractmethod
    def setup_model(self, device, *args, **kwargs):
        raise NotImplemented

    @abc.abstractmethod
    def get_sequence_embeddings(self, sequences):
        """ Expected to return a numpy.ndarray, of shape:
        len(sequences) x (model_embed_seq_len / model_embed_seq_resolution) x model_embed_features

        First dimension is the query sequence, second is the embeddings corresponding to parts of the sequence,
        and third are the embedding features.

        Args:

            sequences (list<str>): List of query sequences to do a forward pass through the model.
        """
        raise NotImplemented


