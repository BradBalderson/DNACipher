from .embedding_model import EmbeddingModel

import torch

import borzoi_pytorch

class BorzoiEmbed(EmbeddingModel):

    """ Class that can generate sequence embeddings from Enformer, which is child of EmbeddingModel.
    """

    model_name = "borzoi"

    def __init__(self, device):
        """ Overrides the EmbeddingModel constructor, so don't need as much input!
        """
        super().__init__(device, 524_288, 196_608, 32, 1920)

    def setup_model(self, device):
        """ Setups up the Enformer model.
        """
        from borzoi_pytorch import Borzoi
        model = Borzoi.from_pretrained('johahi/borzoi-replicate-0').to( device )
        return model

    @staticmethod
    def seq_encoder( dna_str ):
        dna_lkp = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        seq_ohe = torch.zeros((1, 4, len(dna_str)), dtype=torch.float32)
        for i, s in enumerate(dna_str):
            s_upper = s.upper()
            if s_upper in dna_lkp: # Every other character put as a 0, particularly N content.

                seq_ohe[0, dna_lkp[s_upper], i] = 1

        return seq_ohe

    def get_sequence_embeddings(self, sequences, output='numpy'):
        """ Return Enformer embeddings of the inputted sequences.
        """

        #### Converting the sequences to a tensor representation.
        # TODO make this a OHE encoding...
        # This location shows how to do the expected OHE correctly:
        # https://github.com/yangzhao1230/Enformer_Borzoi_Training_Pytorch/blob/main/dataloaders/npy_dataset.py
        seq_tensors = torch.concat( [BorzoiEmbed.seq_encoder(seq_) for seq_ in sequences] )

        #### Extracting the embeddings.
        with torch.no_grad():
            _, embeddings = self.model(seq_tensors.to( self.model.device ), return_embeddings=True)

        #### Returning the embeddings.
        # Old return, decided against this.
        #return embeddings.to('cpu').numpy()
        return embeddings.permute(0,2,1) # So is the expected seq, pos, embedding output.

