from .embedding_model import EmbeddingModel

import torch

from enformer_pytorch import from_pretrained

class EnformerEmbed(EmbeddingModel):

    """ Class that can generate sequence embeddings from Enformer, which is child of EmbeddingModel.
    """

    model_name = "enformer"

    def __init__(self, device):
        """ Overrides the EmbeddingModel constructor, so don't need as much input!
        """
        super().__init__(device, 196_608, 114_688, 128, 3072)

    def setup_model(self, device):
        """ Setups up the Enformer model.
        """
        model = from_pretrained('EleutherAI/enformer-official-rough').to( device )
        return model

    @staticmethod
    def seq_encoder( dna_str ):
        dna_lkp = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

        encoded_seq = []
        for s in dna_str:
            encoded_seq.append( dna_lkp[s.upper()] )

        return torch.tensor( encoded_seq )

    def get_sequence_embeddings(self, sequences, output='numpy'):
        """ Return Enformer embeddings of the inputted sequences.
        """

        #### Converting the sequences to a tensor representation.
        seq_tensors = torch.concat( [EnformerEmbed.seq_encoder(seq_).unsqueeze(0) for seq_ in sequences] )

        #### Extracting the embeddings.
        with torch.no_grad():
            _, embeddings = self.model(seq_tensors.to( self.model.device ), return_embeddings=True)

        #### Returning the embeddings.
        # Old return, decided against this.
        #return embeddings.to('cpu').numpy()
        return embeddings

