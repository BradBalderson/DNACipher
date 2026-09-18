from collections import defaultdict

import sys

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from torch.utils.data import Dataset

import pyranges as pr

from .embedding_model import EmbeddingModel

class EmbeddingGenerator(Dataset):
    # TODO re-write this description, have made a simpler version which makes it so only need to implement a few simple
    #  things in the EmbeddingModel class, and then this class handles all of the other tasks around getting specific
    #  embeddings for specific queries.
    """ Parent class of an EmbeddingGenerator, which generates embeddings of a DNA sequence. The DNA sequence is
     specified by an input reference genome, from which positions in that genome can be queried to generate embeddings.
     The class itself handles of the queries from the reference genome according to the embedding models input size, and
     also when querying specific positions in the reference genome it will not query centred on the position, but will
     generate a larger tiled embedding from the reference genome, and subset the embeddings from that tile to get the
     query position embedding. This approach is useful for downstream model training, because it means the new model
     can handle embeddings from any arbitrary position in the sequence, and also when doing batch querying it enables
     faster generation of embeddings (because you don't have to generate a new larger embedding for every query, but
     instead inference once and extract multiple embeddings from the one forward pass of the model).

     e.g. for enformer:

        x = EmbeddingGenerator(fasta_file, 196_608,  114_688, 128, 3072, 0)
        x.create_genome_tiles()

        # Checking these in IGV, to make sure our embeddings can cover the full genome.
        for i in range(1,23):
            x.chr_preds[f"chr{i}"].to_csv(f'~/Downloads/chr{i}_pred_tiles.bed', sep='\t', header=None, index=False)

     Any inheriting child class will represent a single model that can generate embeddings, and so must implement the
     following functions to handle the specific model:

        1. setup_model -> uses **kwargs input to the class initialization, to setup the model and attach it to this
                                                  class so that it can be used internally by the get_embedding function.
        2. get_sequence_embedding -> which must take as input a DNA sequence as a string, and return the embedding.
    """

    def __init__(self, embed_model, query_positions, fasta_file, query_embed_resolution=0, verbose=True):
        """ Creates an EmbeddingGenerator.

        Args:
            fasta_file (str): Path to the fasta file to load the sequences from.

            embed_model (EmbeddingModel): An embedding model instance, from which will generate the embeddings.

            max_batch_size (int): Refers to the number of tiling sequences which can be generated in one forward pass through
                                    the model with one query. This should be set depending on the available GPU RAM.

            query_embed_resolution (int): For any positional query for an embedding, the desired resolution of that query.
                                    For example, if input chr1:400 as the query, with a query_embed_resolution of 150 bp,
                                    then the embedder will return the average of the embeddings features, for sequence embeddings
                                    that intersect the position by +/-150bp. Enformer as an example, with a 150 bp query,
                                    will take the average of 3 embeddings, that intersect positions chr1:250 to chr1:550.
                                    If this is set to 0, it just takes the embedding that intersects the query position.
        """

        if not issubclass(type(embed_model), EmbeddingModel):
            raise Exception(f"Need to input an instance of the EmbeddingModel class as the embed_model input. "
                            f"Got {type(embed_model)}.")

        self.embed_model = embed_model

        self.query_positions = query_positions
        self.query_embed_resolution = query_embed_resolution

        # Creating a new set of positions, +/- xbp from the input positions, so can get the indices of the outputs!
        if type(query_embed_resolution) != type(None): # Assumes the query positions are a single position !!
            chroms, pos = self.query_positions[:,0], self.query_positions[:,1]
            starts, ends = pos-self.query_embed_resolution, pos+self.query_embed_resolution

        else: # This means the inputs is a bed file, so we don't have a fixed length either side of a position.
            chroms, starts, ends = self.query_positions[:,0], self.query_positions[:,1], self.query_positions[:,2]

        self.query_ranges = np.array([chroms, starts, ends]).transpose()

        # attaching the genome data.
        self.genome_data = Fasta(fasta_file, as_raw=True, rebuild=False)

        ### Checking the query position chromsomes and the input fasta file chromosomes:
        genome_chroms = set([chr_ for chr_, chr_fasta_record in self.genome_data.items()])
        chroms_per_query = query_positions[:,0]
        query_chroms = set(list(chroms_per_query))

        missing_chroms = query_chroms.difference( genome_chroms )
        if len(missing_chroms) > 0:
            err_str = "Query position chromosome not represented in input fasta.\n" +\
                      f"Query chroms: {query_chroms}\n" +\
                      f"Fasta chroms: {genome_chroms}\n"+\
                      f"Missing chroms: {missing_chroms}"
            raise Exception( err_str )
        else:
            # Set the chromosomes represented
            self.chroms = list( query_chroms )
            print( self.chroms )

        # precomputing the tiled regions for this genome.
        if verbose:
            print("Setting up embedding tiles..", file=sys.stdout, flush=True)
        self.chr_tiles, self.chr_preds, self.n_tiles = self.create_genome_tiles()
        #self.chr_tiles[self.chroms[0]].to_csv("tiles.bed", sep='\t', header=None, index=None)
        if verbose:
            print(f"FINISHED. Total of {self.n_tiles} tiles across {len(self.chroms)} chroms", file=sys.stdout, flush=True)

        # Creating a mapping from the query positions to the precomputed genome tiles and their output embedding
        # positions, so that can quickly extract the relevant positions!
        # Attaching this information so can easily determine which tiles to generate per query.
        if verbose:
            print("Mapping query positions to embedding tiles...", file=sys.stdout, flush=True)
        self.query_to_chr_tileindex, self.query_to_tileindex_topred = self.map_positions_to_tiles_and_preds()
        if verbose:
            print("Finished query -> tile mapping.", file=sys.stdout, flush=True)

    def __len__(self):
        return self.query_positions.shape[0]

    def create_genome_tiles(self):
        """ Creates genome tiles, that tiles across the genome with the input sequence size of the model, and
            these are used to define the sequences from which particular queries will be extracted.

            Need to make these tiles per chromosome, and handle the chromosome edges appropriately. Namely,
            for the start of the chromosome, because there will be a lack of sequence context to the LEFT of the
            chromosome, then will need to add 'N' padding to account for this. Similarly for the end of the chromosome.
        """

        chr_tiles = {} # Defines the input sequences.
        chr_preds = {} # Defines the corresponding covered sequences embeding predictions for each input sequence.

        #### Need to make these per chromsome
        #for chr_, chr_fasta_record in self.genome_data.items():
        for chr_ in self.chroms:

            chr_tilei = 0

            chr_fasta_record = self.genome_data[chr_]

            chr_tiles[chr_] = []
            chr_preds[chr_] = []

            chrom_size = len(chr_fasta_record)

            ## For the first tile, it will need to account for the missing sequence to the LEFT:
            # So this will be missing the left context, but will handle that when we generate the embeddings!
            start_ = 0
            end_ = self.embed_model.embed_seq_len + self.embed_model.edge_context
            ### Handle a chromosome shorter than the model input sequence!
            small_chrom = False
            if end_ >= chrom_size:
                end_ = chrom_size
                small_chrom = True

            pred_start = 0
            pred_end = start_ + self.embed_model.embed_seq_len

            # NOTE that indexing in the fasta file is 1-based, and incluse of the last part of the slice.
            # so this yields as sequence of length 2001: self.genome_data.get_seq("chr1", 10_000, 12_000)
            # Will handle this in the get_tile_seq function, and use standard Pythonic indexing here.
            first_tile = [chr_, start_, end_, int(chr_tilei)]
            chr_tiles[chr_].append( first_tile )

            first_pred = [chr_, pred_start, pred_end, int(chr_tilei)]
            chr_preds[chr_].append( first_pred )

            predictable_bases = (pred_end-pred_start)+2 #+2 accounts for the INCLUSIVE sequence reading of the end_ and start_ position.

            chr_tilei += 1

            if small_chrom:
                continue

            # Now going through and adding tiles from this.
            while end_ < chrom_size:
                start_ = pred_end - self.embed_model.edge_context # Need these extra LEFT context so can make embeddings for ALL the genome.
                end_ = start_ + self.embed_model.input_seq_len # And then this adds the embed_len, and then the addition RIGHT context.

                pred_start = start_ + self.embed_model.edge_context
                pred_end = end_ - self.embed_model.edge_context

                # Off the edge of the chromosome, so now just truncate the end, and will handle adding N-padding later.
                if end_ > chrom_size:
                    end_ = chrom_size

                new_tile = [chr_, start_, end_, int(chr_tilei)]
                chr_tiles[chr_].append( new_tile )

                new_pred = [chr_, pred_start, pred_end, int(chr_tilei)]
                chr_preds[chr_].append( new_pred )

                predictable_bases += (pred_end-pred_start) + 2

                chr_tilei += 1

            # The last tile needs to account for missing sequence to the right, which is trying because
            # this CAN result in two INTERSECTING tiles that correspond to a query position.
            # Will handle this at embedding time,
            # if a query position intersects two end tiles, then will use the tile for that query
            # which has the greatest sequence context on either side of the query position.
            start_ = chrom_size - (self.embed_model.embed_seq_len + self.embed_model.edge_context)
            last_tile = [chr_, start_, chrom_size, int(chr_tilei)]
            chr_tiles[chr_].append( last_tile )

            # And the last positions covered by this last tile.
            chr_preds[chr_].append( [chr_, start_+self.embed_model.edge_context, chrom_size, int(chr_tilei)] )

            chr_tilei += 1

        # Converting into bed dataframes...
        chr_tiles_dfs, chr_preds_df = {}, {}
        n_tiles = 0
        for key, values in chr_preds.items():

            chr_tiles_dfs[key] = pd.DataFrame( chr_tiles[key], columns=["Chromosome", "Start", "End", "Name"] )
            chr_preds_df[key] = pd.DataFrame( values, columns=["Chromosome", "Start", "End", "Name"] )

            n_tiles += chr_preds_df[key].shape[0]

        return chr_tiles_dfs, chr_preds_df, n_tiles

    def map_positions_to_tiles_and_preds(self):
        """ Creates a mapping between the input query sequences, the tiles, and the outputs from the tiles, so can just
        precompute the embedding tiles and extract the query regions from them.
        """

        ##### Getting the nearest tiles to each query per chromosome.
        ##### Determining the query regions:
        # Creating a new set of positions, +/- xbp from the input positions, so can get the indices of the outputs!
        chroms, starts, ends = self.query_ranges[:,0], self.query_ranges[:,1], self.query_ranges[:,2]
        indices_ = np.array( list(range(len(starts))) )

        #### Now going through and determining the appropriate tile for each:
        query_to_chr_tileindex = {}
        query_to_tileindex_topred = {}
        for chrom in self.chroms:

            chr_pred_ranges = pr.PyRanges( self.chr_preds[chrom] )

            chrom_queries = np.where( chroms==chrom )[0]

            chrom_query_df = pd.DataFrame([chroms[chrom_queries],
                                           starts[chrom_queries], ends[chrom_queries], indices_[chrom_queries]],
                                                             index=['Chromosome', "Start", "End", "Name"]).transpose()
            chrom_query_ranges = pr.PyRanges( chrom_query_df )

            # This might not neccesarily get the tile with the highest overlap, just the closest. Will try a different method.
            #closest_tiles = chrom_query_ranges.nearest(chr_pred_ranges).df

            def highest_overlap(gr1, gr2):
                # 1. Join to find all overlaps
                # By default, join uses intersection.
                joined = gr1.join(gr2, suffix="_b")

                # The resulting dataframe will have "Start" and "End" (from gr1)
                # and "Start_b" and "End_b" (from gr2).
                # The intersection start is max(Start, Start_b) and end is min(End, End_b).

                # 2. Calculate intersection length
                joined_df = joined.df
                joined_df['intersect_len'] = (
                        joined_df[['End', 'End_b']].min(axis=1) -
                        joined_df[['Start', 'Start_b']].max(axis=1)
                )

                # 3. Keep the max intersection per query interval (ID)
                best_intersections = joined_df.sort_values('intersect_len', ascending=False).drop_duplicates('Name')

                return best_intersections

            closest_tiles = highest_overlap(chrom_query_ranges, chr_pred_ranges)

            print(closest_tiles, file=sys.stdout, flush=True)

            # This the number of embeddings returned along the sequence, if we have a query which extends over
            # a tile, we will simply drop off any embeddings that don't intersect that tile.
            max_pos_embeds = self.embed_model.embed_seq_len // self.embed_model.embed_seq_resolution

            ## Now getting the bins within each tile!
            for row_index, row_data in closest_tiles.iterrows():

                query_index = int( row_data["Name"] )
                tile_index = int( row_data["Name_b"] )
                query_start, query_end = row_data["Start"], row_data["End"]
                pred_start, pred_end = row_data['Start_b'], row_data['End_b']

                start_dist_from_tile_start = query_start - pred_start
                end_dist_from_tile_start = query_end - pred_start

                # The min and the max accounts for the query region intersecting with the edges of the tile - just use
                # the regions which overlap with the tile itself.
                start_bin = start_dist_from_tile_start // self.embed_model.embed_seq_resolution
                end_bin = end_dist_from_tile_start // self.embed_model.embed_seq_resolution

                start_bin = max([0, start_bin])
                end_bin = min([max_pos_embeds-1, end_bin])

                tile_pred_intersect_indices = np.array(list(range(start_bin, end_bin+1)))

                # Also calculating the positions of these bins to make sure they make sense:
                pred_bin_start = (start_bin * self.embed_model.embed_seq_resolution) + pred_start
                pred_bin_end = ((end_bin+1) * self.embed_model.embed_seq_resolution) + pred_start

                ### Error checking the embeddings intersect he query
                query_intersects = query_start >= pred_bin_start or query_end <= pred_bin_end
                if not query_intersects:
                    raise Exception(
                        f"Detect incorrect overlap of query with tiled embedding output."+\
                        f"query region: {chrom}:{pred_start}-{pred_end}"+\
                        f"pred region: {chrom}:{pred_bin_start}-{pred_bin_end}"
                    )

                query_to_tileindex_topred[query_index] = {'chr': chrom,
                                                  'tile_index': tile_index,
                                                  'tile_position': (pred_start, pred_end),
                                                  'pred_indices': tile_pred_intersect_indices,
                                                  'pred_starts': pred_bin_start,
                                                  'pred_ends': pred_bin_end,
                                                  }
                query_to_chr_tileindex[query_index] = (chrom, tile_index)

        return query_to_chr_tileindex, query_to_tileindex_topred

    def get_tile_seq(self, chr_, tile_index):
        """ Retrieves the sequence of a chromosome tile, where tile_index refers to the tile in self.chr_tiles[chr].iloc[idx,:].
        """

        start_, end_ = list(self.chr_tiles[chr_].iloc[tile_index, 1:3])

        # NOTE the coordinates system used here is NOT Pythonic: 1-based, closed interval.
        # Closed interval means INCLUSIVE of the start and end points, whereas Python drops the end point.
        # So adjusting the coordinates from the stored Python 0-based, EXCLUSIVE of the end-point to this coordinate
        # system to get the expected sequence:
        tile_seq = self.genome_data.get_seq(chr_, start_+1, end_)
        seq_len = len(tile_seq)

        if seq_len != self.embed_model.input_seq_len:
            if seq_len > self.embed_model.input_seq_len:
                raise Exception(f"Something wrong with tiling, got seqlen: {seq_len} for tile: {chr_, start_, end_}")

            seq_pad = 'N' * (self.embed_model.input_seq_len - seq_len)
            if start_ == 1: # First tile, so need to add left N padding to make the difference.
                tile_seq = seq_pad + tile_seq
            else: # Has to be one of the last tiles, so add right N padding.
                tile_seq = tile_seq + seq_pad

            new_len = len(tile_seq)
            if not new_len == self.embed_model.input_seq_len:
                raise Exception(f"Something wrong with tiling, AFTER padding, got seqlen: {new_len} for tile: {chr_, start_, end_}")

        return tile_seq

    def __getitem__(self, idx):
        """ Retrieves the embeddings for the self.query_positions[idx].
        """

        # Determining the kind of index inputted:
        if type(idx)==tuple:
            raise Exception("Column indexing not supported.")

        #### Dealing with format of the region indices
        if type(idx) == slice:  # Not supported
            raise Exception("Slicing regions not supported.")

        elif type(idx) == int:  # Supported, but convert to list for simplicity.
            idx = [idx]

        elif type(idx) != list:  # Only support list indexing.
            raise Exception(f"idx of type {type(idx)} not supported.")

        ## Now determine what tiles need to be computed to get the embeddings for the query regions.
        queried_pos_chr_tileindex = [self.query_to_chr_tileindex[ query_index ] for query_index in idx]
        tiles_queried = list( set( queried_pos_chr_tileindex ) )

        ## Retrieving the sequences of these tiles:
        tile_sequences = [self.get_tile_seq(chr_, tile_index) for chr_, tile_index in tiles_queried]

        ## Now getting the embeddings for these tiles:
        tile_embeddings = self.embed_model.get_sequence_embeddings( tile_sequences ).to('cpu').numpy()

        ## Now getting the embeddings for the query positions!
        # Getting a mapping from the queried tiles to the computed tile embedding matrix.
        query_pos_embeddings = np.zeros((len(queried_pos_chr_tileindex), self.embed_model.embed_features))

        for queryi, (query_index, chr_tileindex) in enumerate( zip(idx, queried_pos_chr_tileindex) ):

            tile_embedding_index = tiles_queried.index( chr_tileindex )
            pred_embedding_indices = self.query_to_tileindex_topred[ query_index ][ 'pred_indices' ]

            # Sanity checking that the mapping is consistent:
            chr_, tile_ = chr_tileindex
            start_ = self.chr_preds[chr_].iloc[tile_, :]['Start']
            pred_start = start_+(self.embed_model.embed_seq_resolution*pred_embedding_indices[0])
            pred_end = start_+(self.embed_model.embed_seq_resolution*pred_embedding_indices[-1])

            query_chr, query_pos = self.query_positions[query_index, [0, 1]]

            # Checking the query tile location intersects with the query location, check that the generator data is
            # internally consistent.
            if not chr_ == query_chr or not (query_pos >= pred_start or query_pos <= pred_end):
                raise Exception(f"Tile query error, requested {query_chr}:{query_pos} embed, but got non-intersecting "
                                f"tile embed {chr_}:{pred_start}-{pred_end}"
                                )

            # Getting the mean embeddings across the outputs embeddings of the tile the intersect the query region!
            query_embedding = tile_embeddings[tile_embedding_index, pred_embedding_indices, :].mean( axis=0 )
            query_pos_embeddings[queryi, :] = query_embedding

        return {}, query_pos_embeddings
