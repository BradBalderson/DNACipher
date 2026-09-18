""" Contains basic tests for running the key analysis through a debugger to check all variables are as expected.

To set this up in the debugger in PyCharm, do this:

    Run -> Edit Configurations
    Name = "run_tests"
    Select the drop-down box that currently says "script", and change to "module"
    Set to dnacipher_train.test
    Set the working directory to the location of the folder "DNACipher_Training/"
    Click "Apply" then "OK"

    Now to run the Debugger, "Run" -> "Debug" -> "run_tests"

    That will run this script in the debugger, keeping the relative imports working correctly.
"""

import sys
from pathlib import Path

test_file_path = Path(__file__).parent # Path to this test file, so can get the DNACipher_train package scripts.

sys.path.append( test_file_path )

test_data_dir = f"{test_file_path}/../example_data/small_dataset/"
test_full_dir = f"{test_file_path}/../example_data/full_dataset/"
test_borzoi_dir = f"{test_full_dir}borzoi/"

def test_embeds():

    from .embed import run_embed_gen as reg

    # reg.generate_embeddings("TEST", f"{test_data_dir}PRECOMP-MID-AVG_GENOME_genome_rois.small.txt",
    #                         f"{test_data_dir}../hg38_renamed.fa", 147,
    #                     "enformer", "cpu", 56, test_data_dir,
    #                     )

    ### Testing with chrMT as input!
    reg.generate_embeddings("TEST", f"{test_data_dir}PRECOMP-MID-AVG_GENOME_genome_rois.small.wMT.txt",
                            f"{test_data_dir}../hg38_renamed.fa", 147,
                        "enformer", "cpu", 56, test_data_dir,
                        )

def compare_embeds():
    """Testing that the generated embeddings correlate well to the expected embeddings if we just run the
    the query sequence through the model.
    """

    genome_file = f"{test_full_dir}PRECOMP-MID-AVG_GENOME_genome_rois.train_test.100N.split.txt"
    embed_file = f"{test_full_dir}REPROD_EMBED_enformer_embeds.100N.split.h5"
    fasta_file = f"{test_data_dir}../hg38_renamed.fa"
    device = 'cpu'

    import matplotlib.pyplot as plt

    import numpy as np
    import pandas as pd
    from dnacipher_train.helpers.dataset_classes import RegionFeaturesSubset
    from dnacipher_train.embed import md
    import dnacipher_train.embed.embedding_generator as eg
    from pyfaidx import Fasta
    import scipy

    genome_rois = pd.read_csv(f'{genome_file}', sep='\t', header=None)

    # The precomputed embedding data.
    embed_data = RegionFeaturesSubset(genome_rois.values, embed_file)

    # Enformer embedding model
    embed_model = md.model_classes[md.model_names.index('enformer')]( device )

    # Recreating the embedding generator object, that was used to write the above embeddings.
    query_embed_resolution = 147 # Set the same as when I precomputed the embeddings.
    embed_generator = eg.EmbeddingGenerator(embed_model, genome_rois.values, fasta_file, query_embed_resolution)

    # Quick test
    embeds_ = embed_generator[list(range(100, 200, 10))]

    # Sequence generator
    genome_data = Fasta(fasta_file, as_raw=True, rebuild=False)

    # Checking sequences!
    input_len = embed_model.input_seq_len
    side_seq = input_len // 2

    n_checks = genome_rois.shape[0]
    cadence = 15_000 # OK checking across the whole dataset now with a high step size to check across chromosomes!

    check_indices = list(range(0, n_checks, cadence))
    # Debugging an issue when I deploy this solution !!
    #check_indices = list(range(4351-2, 4351+2))

    corrs = np.zeros((len(check_indices), 2))
    for i, index in enumerate( check_indices ):

        chr_, pos, alloc_, _, _ = genome_rois.values[index, :]

        print(f"{i}/{n_checks//cadence}", chr_, pos, alloc_,)

        query_start, query_end = pos-side_seq, pos+side_seq

        # Dealing with sizes, particuarly if the query would go off the edge of the chromosome.
        chr_fasta_record = genome_data[chr_]
        chrom_size = len(chr_fasta_record)

        left_pad = 0 if query_start > 0 else abs(query_start)
        right_pad = 0 if query_end < chrom_size else chrom_size-query_end

        query_start, query_end = max([0, query_start]),  min([chrom_size, query_end])

        # Retrieving the query with padding if off contig edge.
        query_seq = genome_data.get_seq(chr_, query_start + 1, query_end)
        query_seq = ('N' * left_pad) + query_seq + ('N' * right_pad)

        # Querying the sequence through the model
        query_embeds = embed_model.get_sequence_embeddings([query_seq])
        query_embed = query_embeds[0, query_embeds.shape[1]//2, :] # The middle position, since centred on query region.

        # And also computing this using our tiling generator:
        tile_embed = embed_generator[index][1][0, :]

        # Now getting the precomputed embed for comparison
        precomp_embed = embed_data[index][1]

        corrs[i, 0] = scipy.stats.pearsonr(query_embed, precomp_embed)[0]
        corrs[i, 1] = scipy.stats.pearsonr(query_embed, tile_embed)[0]
        #corrs[i, 1] = scipy.stats.spearmanr(query_embed, precomp_embed)[0]

    plt.scatter(query_embed, precomp_embed)
    plt.xlabel('Example Query embed')
    plt.ylabel('Example Precomp embed')
    plt.show()

    plt.scatter(precomp_embed, tile_embed)
    plt.xlabel('Example Gen embed')
    plt.ylabel('Example Precomp embed')
    plt.show()

    fig, axes = plt.subplots(ncols=2)
    axes = axes.ravel()
    axes[0].hist(corrs[:i,0], bins=20, color='blue', alpha=.8, density=False, label='R precomp')
    axes[1].hist(corrs[:i,1], bins=20, color='orange', alpha=.8, density=False, label='R gen')
    axes[0].legend()
    axes[1].legend()
    plt.show()

    plt.scatter(genome_rois.values[:i, 1], corrs[:i, 0])
    plt.xlabel('Genome position along chr1')
    plt.ylabel('R precomp vs query')
    plt.show()

    plt.scatter(genome_rois.values[:i, 1], corrs[:i, 1])
    plt.xlabel('Genome position along chr1')
    plt.ylabel('R gen vs query')
    plt.show()

    """OK this looks great now! The embeddings are very solid!"""

    print("here")

def test_signals():

    from .signals import signal_processing as sp

    # sp.run_signal_precomputation("TEST_signals",
    #                              f"{test_data_dir}PRECOMP-MID-AVG_GENOME_genome_rois.small.txt",
    #                              f"{test_data_dir}peak_bigwigs/",
    #                              f"{test_data_dir}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.small.tsv",
    #                              100, 1, 147, test_data_dir)

    ### New version I am testing with deep tools!
    sp.run_signal_precomputation("TEST_signals",
                                 f"{test_data_dir}PRECOMP-MID-AVG_GENOME_genome_rois.small.txt",
                                 f"{test_data_dir}peak_bigwigs/",
                                 f"{test_data_dir}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.small.tsv",
                                 10, 1, 147, test_data_dir, method="deeptools")

    # TODO should also get the newest version that uses bwtool up and running..

def test_norm():

    from .signals import signal_normalization as sn

    sn.fit_signal_normalization("TEST_full_norm",
                                f"{test_full_dir}PRECOMP-MID-AVG_GENOME_genome_rois.train_test.100N.split.txt",
                                f"{test_full_dir}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.tsv",
                                f"{test_full_dir}REPROD_BWTOOL_epivalues.100N.split.h5", test_full_dir)

    # sn.fit_signal_normalization("TEST_norm", genome_file, sample_file, signal_file, out_dir)
    #
    # norm_file = out_dir + f'{run_name}_norm_stats.pkl'
    # sn.apply_signal_normalization(run_name, genome_file, sample_file, signal_file, norm_file, out_dir)

def test_train():

    from .train import train_model as tm

    train_borzoi = True
    small_dataset = False
    if small_dataset and not train_borzoi:
        run_name = "SMALL_TEST_TRAIN"
        genome_file = f"{test_data_dir}PRECOMP-MID-AVG_GENOME_genome_rois.small.10N.split.txt"
        sample_file = f"{test_data_dir}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.small.tsv"
        signal_file = f"{test_data_dir}SMALL_BWTOOL_epivalues.normalised.10N.split.h5"
        embed_file = f"{test_data_dir}SMALL_EMBED_enformer_embeds.10N.split.h5"
        out_dir = test_data_dir

    else: # full_dataset
        run_name = "FULL_TEST_TRAIN"
        genome_file = f"{test_full_dir}PRECOMP-MID-AVG_GENOME_genome_rois.train_test.100N.split.txt"
        sample_file = f"{test_full_dir}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.tsv"
        signal_file = f"{test_full_dir}REPROD_BWTOOL_epivalues.normalised.100N.split.h5"
        embed_file = f"{test_full_dir}REPROD_EMBED_enformer_embeds.100N.split.h5"
        out_dir = test_full_dir

    if train_borzoi:
        run_name = "BORZOI_TEST_TRAIN"
        embed_file = f"{test_full_dir}BORZOI_EMBED_borzoi_embeds.100N.split.h5"

    # training params that need to change depending on machine...
    batch_size = 256
    epochs_warmup = 1
    epochs_final = 1
    cpu = 5
    device = 'mps:0'
    eval_batches = 2

    model_config_file = None

    tm.train_dnacipher_model(run_name, genome_file, sample_file, signal_file, embed_file, out_dir,
                             model_config_file,
                          # train parameters, need to change depending on resources.
                          batch_size, cpu, device, epochs_warmup, epochs_final, eval_batches,
                             n_regions=int(batch_size*4) # For testing purposes.
                            )

def test_eval():
    """Testing running eval-only with a pretrained model - specifically the final pre-trained model.
    """
    run_name = "FULL_EVAL"
    genome_file = f"{test_full_dir}PRECOMP-MID-AVG_GENOME_genome_rois.train_test.100N.split.txt"
    sample_file = f"{test_full_dir}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.tsv"
    signal_file = f"{test_full_dir}REPROD_BWTOOL_epivalues.normalised.100N.split.h5"
    embed_file = f"{test_full_dir}REPROD_EMBED_enformer_embeds.100N.split.h5"
    out_dir = test_full_dir

    weights_dir = "/Users/bradbalderson/Desktop/projects/myPython/DNACipher_Training/dnacipher_train/weights/"
    weights_path = f"{weights_dir}TRAINING_DNACV5_MID-AVG-GENOME_ORIG-ALLOC_ENFORMER0_FINETUNE_STRATMSE_model_weights.pth"
    model_config_file = None

    device = 'mps:0'
    batch_size = 64
    eval_batches = 10
    cpu = 1

    from .train import train_eval as te

    te.eval(run_name, genome_file, sample_file, signal_file, embed_file,
            weights_path, out_dir, model_config_file,
            # train parameters, need to change depending on resources.
            batch_size, cpu, device,eval_batches,
           )

def test_multivar_inference():
    """ Testing if the multivariant effect inference is working correctly.
    """

    from dnacipher_train.infer import run_effect_inference as infer

    run_name = "TEST_INFER"

    data_dir = "/Users/bradbalderson/Desktop/projects/myPython/DNACipher_Training/example_data/infer/"
    out_dir = f"{data_dir}infer_out/"

    vcf_path = f"{data_dir}gtex_variants_SMALL.vcf"

    celltypes = f"{data_dir}celltypes.txt"
    assays = f"{data_dir}assays.txt"

    fasta_file_path = f"{data_dir}../hg38_renamed.fa"

    # For specifying the DNACipher model.
    sample_file = None
    model_config_file = None
    weights_path = None

    # For specifying the embedding model.
    model_name = "enformer"

    # General
    device = "cpu"

    index_base = 1

    correct_ref = False

    seq_pos_col = None
    effect_region_start_col = None
    effect_region_end_col = None
    batch_size = None
    batch_by = None

    all_combinations = True

    scoring_method = "sum_logfc"

    verbose = True

    infer.infer_multivariant_effects(run_name, vcf_path, celltypes, assays, fasta_file_path,
                                     sample_file, model_config_file, weights_path, model_name,
                                     device, index_base,
                                     correct_ref, seq_pos_col, effect_region_start_col, effect_region_end_col,
                                     batch_size,
                                     batch_by, all_combinations, scoring_method, out_dir, verbose,
                                     n_vars=2,
                                     )

def test_ipscore_effect():
    """ Testing if the multivariant effect inference is working correctly.
    """

    from dnacipher_train.infer import run_effect_inference as infer

    run_name = "TEST_IPSCORE"

    data_dir = '/Users/bradbalderson/Desktop/projects/MRFF/data/seqcipher/revision/xQTLs/'
    data_dir3 = '/Users/bradbalderson/Desktop/projects/MRFF/data/seqcipher/revision/borzoi/'
    out_dir = data_dir

    out_prefix = f"{out_dir}{run_name}_"

    vcf_path = f"{data_dir}ipscore_xqtl_queries.renamed.txt"

    celltypes = f"{data_dir}ipscore_celltypes.txt"
    assays = f"{data_dir}ipscore_assays.txt"

    fasta_file_path = "/Users/bradbalderson/Desktop/projects/myPython/Sheriff/example_data/hg38.fa"

    # For specifying the DNACipher model.
    sample_file = f"{data_dir3}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_0.tsv"
    model_config_file = None
    weights_path = f"{data_dir3}BORZOI_TRAIN_0_model_weights.pth"

    # For specifying the embedding model.
    model_name = "borzoi"

    # General
    device = "cpu"

    index_base = 1

    correct_ref = False

    seq_pos_col = "SEQ_POS"
    effect_region_start_col = "START_EFFECT"
    effect_region_end_col = "END_EFFECT"
    batch_size = 3
    batch_by = None

    all_combinations = True

    scoring_method = "sum_logfc"

    verbose = True

    infer.infer_multivariant_effects(out_prefix, vcf_path, celltypes, assays, fasta_file_path,
                                     sample_file, model_config_file, weights_path, model_name,
                                     device, index_base,
                                     correct_ref, seq_pos_col, effect_region_start_col, effect_region_end_col,
                                     batch_size,
                                     batch_by, all_combinations, scoring_method, verbose, n_vars=2
                                     )

def compare_borzoi_embeds():
    """Testing that the generated embeddings correlate well to the expected embeddings if we just run the
    the query sequence through the model.
    """

    genome_file = f"{test_full_dir}PRECOMP-MID-AVG_GENOME_genome_rois.train_test.100N.split.txt"
    #embed_file = f"{test_full_dir}REPROD_EMBED_enformer_embeds.100N.split.h5"
    fasta_file = f"{test_data_dir}../hg38_renamed.fa"
    device = 'cpu'

    import matplotlib.pyplot as plt

    import numpy as np
    import pandas as pd
    from dnacipher_train.helpers.dataset_classes import RegionFeaturesSubset
    from dnacipher_train.embed import md
    import dnacipher_train.embed.embedding_generator as eg
    from pyfaidx import Fasta
    import scipy

    genome_rois = pd.read_csv(f'{genome_file}', sep='\t', header=None)

    # The precomputed embedding data.
    #embed_data = RegionFeaturesSubset(genome_rois.values, embed_file)

    # Borzoi embedding model
    embed_model = md.model_classes[md.model_names.index('borzoi')]( device )

    # Recreating the embedding generator object, that was used to write the above embeddings.
    query_embed_resolution = 147 # Set the same as when I precomputed the embeddings.
    embed_generator = eg.EmbeddingGenerator(embed_model, genome_rois.values, fasta_file, query_embed_resolution)

    # Quick test
    embeds_ = embed_generator[list(range(100, 200, 10))]

    # Sequence generator
    genome_data = Fasta(fasta_file, as_raw=True, rebuild=False)

    # Checking sequences!
    input_len = embed_model.input_seq_len
    side_seq = input_len // 2

    n_checks = genome_rois.shape[0]
    cadence = 15_000 # OK checking across the whole dataset now with a high step size to check across chromosomes!

    check_indices = list(range(0, n_checks, cadence))
    # Debugging an issue when I deploy this solution !!
    #check_indices = list(range(4351-2, 4351+2))

    corrs = np.zeros((len(check_indices), 2))
    for i, index in enumerate( check_indices ):

        chr_, pos, alloc_, _, _ = genome_rois.values[index, :]

        print(f"{i}/{n_checks//cadence}", chr_, pos, alloc_,)

        query_start, query_end = pos-side_seq, pos+side_seq

        # Dealing with sizes, particuarly if the query would go off the edge of the chromosome.
        chr_fasta_record = genome_data[chr_]
        chrom_size = len(chr_fasta_record)

        left_pad = 0 if query_start > 0 else abs(query_start)
        right_pad = 0 if query_end < chrom_size else chrom_size-query_end

        query_start, query_end = max([0, query_start]),  min([chrom_size, query_end])

        # Retrieving the query with padding if off contig edge.
        query_seq = genome_data.get_seq(chr_, query_start + 1, query_end)
        query_seq = ('N' * left_pad) + query_seq + ('N' * right_pad)

        # Querying the sequence through the model
        query_embeds = embed_model.get_sequence_embeddings([query_seq])
        query_embed = query_embeds[0, query_embeds.shape[1]//2, :] # The middle position, since centred on query region.

        # And also computing this using our tiling generator:
        tile_embed = embed_generator[index][1][0, :]

        # Now getting the precomputed embed for comparison
        #precomp_embed = embed_data[index][1]

        #corrs[i, 0] = scipy.stats.pearsonr(query_embed, precomp_embed)[0]
        corrs[i, 1] = scipy.stats.pearsonr(query_embed, tile_embed)[0]
        #corrs[i, 1] = scipy.stats.spearmanr(query_embed, precomp_embed)[0]

    plt.scatter(query_embed, tile_embed)
    plt.xlabel('Example Query embed')
    plt.ylabel('Example Tile embed')
    plt.show()

    # plt.scatter(query_embed, precomp_embed)
    # plt.xlabel('Example Query embed')
    # plt.ylabel('Example Precomp embed')
    # plt.show()
    #
    # plt.scatter(precomp_embed, tile_embed)
    # plt.xlabel('Example Gen embed')
    # plt.ylabel('Example Precomp embed')
    # plt.show()

    fig, axes = plt.subplots(ncols=2)
    axes = axes.ravel()
    axes[0].hist(corrs[:i,0], bins=20, color='blue', alpha=.8, density=False, label='R precomp')
    axes[1].hist(corrs[:i,1], bins=20, color='orange', alpha=.8, density=False, label='R gen')
    axes[0].legend()
    axes[1].legend()
    plt.show()

    plt.scatter(genome_rois.values[:i, 1], corrs[:i, 0])
    plt.xlabel('Genome position along chr1')
    plt.ylabel('R precomp vs query')
    plt.show()

    plt.scatter(genome_rois.values[:i, 1], corrs[:i, 1])
    plt.xlabel('Genome position along chr1')
    plt.ylabel('R gen vs query')
    plt.show()

    """OK this looks great now! The embeddings are very solid!"""

    print("here")

def trait_impact_map():
    """ Testing new functionality where the inputted variants represent variant effects across all loci of a trait,
    rather than
    """
    import pandas as pd

    import dnacipher_train.infer.deep_variant_impact_mapping as dvim

    data_dir = "/Users/bradbalderson/Desktop/projects/MRFF/data/seqcipher/revision/multi_trait/"

    trait_ = 'T2D'
    selected_gwas_stats_path = f"{data_dir}{trait_}.concat_stats.txt"
    pred_effects_path = f"{data_dir}{trait_}_effects._var_context_effects.txt.gz"

    selected_gwas_stats = pd.read_csv(selected_gwas_stats_path, sep='\t')
    selected_pred_effects = pd.read_csv(pred_effects_path, sep='\t')

    selected_gwas_stats, sig_effects, foldchange_effects, boot_pvals_df, boot_counts_df = dvim.impact_map(
                                                                             selected_gwas_stats, selected_pred_effects,
                                                                    cpu=3,
    )

    print("DONE.")

def test_new_eval():
    """Adding new features to the evaluation, namely determining the KNN cell types within a batch, and using that
        as the mean estimate...
    """

    run_name = "TEST_KNN_CELLTYPE_EVAL"

    fold = 0
    genome_file = f"{test_borzoi_dir}PRECOMP-MID-AVG_GENOME_genome_rois.train_test.borzoi.100N.split.txt"
    sample_file = f"{test_borzoi_dir}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_{fold}.tsv"
    signal_file = f"{test_borzoi_dir}REPROD_BWTOOL_epivalues.BORZOI_NORM_{fold}.normalised.100N.split.h5"
    embed_file = f"{test_borzoi_dir}BORZOI_EMBED_borzoi_embeds.100N.split.h5"
    out_dir = test_borzoi_dir

    weights_path = f"{test_borzoi_dir}BORZOI_TRAIN_{fold}_model_weights.pth"
    model_config_file = None

    ct_knn_json_file = f"/Users/bradbalderson/Desktop/projects/myPython/DNACipher_Training/dnacipher_train/configs/celltype_knn.json"

    device = 'mps:0'
    batch_size = 256
    eval_batches = 10
    cpu = 1

    from .train import train_eval as te

    te.eval(run_name, genome_file, sample_file, signal_file, embed_file,
            weights_path, out_dir, model_config_file, ct_knn_json_file,
            # train parameters, need to change depending on resources.
            batch_size, cpu, device,eval_batches,
           )

def test_infer_signals():
    """Adding new features to the evaluation, namely determining the KNN cell types within a batch, and using that
        as the mean estimate...
    """

    out_dir = '/Users/bradbalderson/Desktop/projects/MRFF/data/seqcipher/revision/performance_compare/'
    run_name = f"{out_dir}TEST_CT_SIGNALS"

    fold = 0
    bed_path = f"{out_dir}celltype_marker_regions.bed"
    celltypes = f"{out_dir}SMALL_train_ct_queries.txt"
    assays = f"{out_dir}SMALL_train_assay_queries.txt"
    sample_file = f"{test_borzoi_dir}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_{fold}.tsv"
    model_name = 'borzoi'

    fasta_file_path = "/Users/bradbalderson/Desktop/projects/myPython/Sheriff/example_data/hg38.fa"

    weights_path = f"{test_borzoi_dir}BORZOI_TRAIN_{fold}_model_weights.pth"
    model_config_file = None

    device = 'cpu' # Won't work on mps:0, because....
    batch_size = 3

    from .infer import run_signal_inference as si

    si.infer_signals(run_name, bed_path, celltypes, assays, fasta_file_path,
                   sample_file, model_config_file, weights_path, model_name, device, batch_size,
                   False, True, n_regions=1
                   )

def test_new_loss():

    from .train import train_model as tm

    train_borzoi = True
    small_dataset = True
    if small_dataset and not train_borzoi:
        run_name = "SMALL_TEST_NEWLOSS"
        genome_file = f"{test_data_dir}PRECOMP-MID-AVG_GENOME_genome_rois.small.10N.split.txt"
        sample_file = f"{test_data_dir}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.small.tsv"
        signal_file = f"{test_data_dir}SMALL_BWTOOL_epivalues.normalised.10N.split.h5"
        embed_file = f"{test_data_dir}SMALL_EMBED_enformer_embeds.10N.split.h5"
        out_dir = test_data_dir

    else: # full_dataset
        run_name = "FULL_TEST_NEWLOSS"
        genome_file = f"{test_full_dir}PRECOMP-MID-AVG_GENOME_genome_rois.train_test.100N.split.txt"
        sample_file = f"{test_full_dir}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.tsv"
        signal_file = f"{test_full_dir}REPROD_BWTOOL_epivalues.normalised.100N.split.h5"
        embed_file = f"{test_full_dir}REPROD_EMBED_enformer_embeds.100N.split.h5"
        out_dir = test_full_dir

    if train_borzoi:
        run_name = "BORZOI_TEST_NEWLOSS"
        embed_file = f"{test_full_dir}BORZOI_EMBED_borzoi_embeds.100N.split.h5"

    # training params that need to change depending on machine...
    batch_size = 64
    epochs_warmup = 3
    epochs_final = 3 # Don't relu the output, in this version will just do softplus training all the way.
    cpu = 5
    device = 'mps:0'
    eval_batches = 20

    model_config_file = f"/Users/bradbalderson/Desktop/projects/myPython/DNACipher_Training/dnacipher_train/configs/softplusLR_ATTN_model_config.yaml"

    tm.train_dnacipher_model(run_name, genome_file, sample_file, signal_file, embed_file, out_dir,
                             model_config_file,
                          # train parameters, need to change depending on resources.
                          batch_size, cpu, device, epochs_warmup, epochs_final, eval_batches,
                             n_regions=int(batch_size*4) # For testing purposes.
                            )

def test_eval_new_loss():
    """Testing the eval with the softplus activation, after doing the training of the model with the softplus output
    activiation, I got nan on all the performance metrics, indicating something went wrong. Am evaluating here if it
    is the EVAL that went wrong with the soft-plus activation.
    """

    run_name = "TEST_BORZOI_SOFTPLUS_EVAL"

    fold = 0
    genome_file = f"{test_borzoi_dir}PRECOMP-MID-AVG_GENOME_genome_rois.train_test.borzoi.100N.split.txt"
    sample_file = f"{test_borzoi_dir}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_{fold}.tsv"
    signal_file = f"{test_borzoi_dir}REPROD_BWTOOL_epivalues.BORZOI_NORM_{fold}.normalised.100N.split.h5"
    embed_file = f"{test_borzoi_dir}BORZOI_EMBED_borzoi_embeds.100N.split.h5"
    out_dir = test_borzoi_dir

    weights_path = f"{test_borzoi_dir}BORZOI_SOFTPLUS_{fold}_model_weights.pth"
    model_config_file = f"/Users/bradbalderson/Desktop/projects/myPython/DNACipher_Training/dnacipher_train/configs/softplus_model_config.yaml"

    ct_knn_json_file = f"/Users/bradbalderson/Desktop/projects/myPython/DNACipher_Training/dnacipher_train/configs/celltype_knn.json"

    device = 'mps:0'
    batch_size = 256
    eval_batches = 30
    cpu = 1

    from .train import train_eval as te

    te.eval(run_name, genome_file, sample_file, signal_file, embed_file,
            weights_path, out_dir, model_config_file, ct_knn_json_file,
            # train parameters, need to change depending on resources.
            batch_size, cpu, device,eval_batches,
           )

def test_infer_signals_attn():
    """Adding new features to the evaluation, namely determining the KNN cell types within a batch, and using that
        as the mean estimate...
    """

    out_dir = '/Users/bradbalderson/Desktop/projects/MRFF/data/seqcipher/revision/performance_compare/'
    run_name = f"{out_dir}TEST_CT_SIGNALS"

    fold = 0
    bed_path = f"{out_dir}celltype_marker_regions.bed"
    celltypes = f"{out_dir}SMALL_train_ct_queries.txt"
    assays = f"{out_dir}SMALL_train_assay_queries.txt"
    sample_file = f"{test_borzoi_dir}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_{fold}.tsv"

    model_name = 'borzoi'

    fasta_file_path = "/Users/bradbalderson/Desktop/projects/myPython/Sheriff/example_data/hg38.fa"

    weights_path = f"/Users/bradbalderson/Desktop/projects/MRFF/data/seqcipher/revision/borzoi/BORZOI_SP_LR_ATTN_{fold}_model_weights.pth"
    model_config_file = f"/Users/bradbalderson/Desktop/projects/myPython/DNACipher_Training/dnacipher_train/configs/softplusLR_ATTN_model_config.yaml"

    device = 'cpu'  # Won't work on mps:0, because....
    batch_size = 3

    from .infer import run_signal_inference as si

    si.infer_signals(run_name, bed_path, celltypes, assays, fasta_file_path,
                     sample_file, model_config_file, weights_path, model_name, device, batch_size,
                     False, True, n_regions=1
                     )

def test_multivar_inference_softplus():
    """ Testing if the multivariant effect inference is working correctly.
    """

    from dnacipher_train.infer import run_effect_inference as infer

    run_name = "TEST_INFER"

    data_dir = "/Users/bradbalderson/Desktop/projects/myPython/DNACipher_Training/example_data/infer/"
    out_dir = f"{data_dir}infer_out/"

    vcf_path = f"{data_dir}gtex_variants_SMALL.vcf"

    celltypes = f"{data_dir}celltypes.txt"
    assays = f"{data_dir}assays.txt"

    fasta_file_path = "/Users/bradbalderson/Desktop/projects/myPython/Sheriff/example_data/hg38.fa"

    # For specifying the DNACipher model.
    fold='0'
    sample_file = f"{test_borzoi_dir}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_{fold}.tsv"
    model_config_file = f"/Users/bradbalderson/Desktop/projects/myPython/DNACipher_Training/dnacipher_train/configs/softplusLR_model_config.yaml"
    weights_path = f"/Users/bradbalderson/Desktop/projects/MRFF/data/seqcipher/revision/borzoi/BORZOI_SOFTPLUS_LR_{fold}_model_weights.pth"

    # For specifying the embedding model.
    model_name = "borzoi"

    # General
    device = "cpu"

    index_base = 1

    correct_ref = False

    seq_pos_col = None
    effect_region_start_col = None
    effect_region_end_col = None
    batch_size = None
    batch_by = None

    all_combinations = True

    scoring_method = "sum_logfc"

    verbose = True

    infer.infer_multivariant_effects(f"{out_dir}{run_name}", vcf_path, celltypes, assays, fasta_file_path,
                                     sample_file, model_config_file, weights_path, model_name,
                                     device, index_base,
                                     correct_ref, seq_pos_col, effect_region_start_col, effect_region_end_col,
                                     batch_size,
                                     batch_by, all_combinations, scoring_method, verbose,
                                     n_vars=2,
                                     )

def investigate_strand_reversal():
    """ Investigating the strand reversal issue I observed when doing the eQTL predictions!
    """

    from dnacipher_train.infer import run_effect_inference as infer

    run_name = "CHECK_STRAND"

    data_dir = "/Users/bradbalderson/Desktop/projects/MRFF/data/seqcipher/revision/gtex_bench/"
    out_dir = f"{data_dir}infer_out/"

    vcf_path = f"{data_dir}gtex_highest_pip_vars.dnac_queries.txt"

    celltypes = f"{data_dir}gtex_query_tissues.txt"
    assays = f"{data_dir}gtex_query_assays.txt"

    fasta_file_path = "/Users/bradbalderson/Desktop/projects/myPython/Sheriff/example_data/hg38.fa"

    # For specifying the DNACipher model.
    fold='0'
    sample_file = f"{test_borzoi_dir}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_{fold}.tsv"
    model_config_file = f"/Users/bradbalderson/Desktop/projects/myPython/DNACipher_Training/dnacipher_train/configs/softplusLR_model_config.yaml"
    weights_path = f"/Users/bradbalderson/Desktop/projects/MRFF/data/seqcipher/revision/borzoi/BORZOI_SOFTPLUS_LR_{fold}_model_weights.pth"

    # For specifying the embedding model.
    model_name = "borzoi"

    # General
    device = "cpu"

    index_base = 1

    correct_ref = False

    seq_pos_col = "SEQ_POS"
    effect_region_start_col = "START_EFFECT"
    effect_region_end_col = "END_EFFECT"
    batch_size = None
    batch_by = None

    all_combinations = True

    scoring_method = "sum_logfc"

    verbose = True

    infer.infer_multivariant_effects(f"{out_dir}{run_name}", vcf_path, celltypes, assays, fasta_file_path,
                                     sample_file, model_config_file, weights_path, model_name,
                                     device, index_base,
                                     correct_ref, seq_pos_col, effect_region_start_col, effect_region_end_col,
                                     batch_size,
                                     batch_by, all_combinations, scoring_method, verbose,
                                     n_vars=2,
                                     )

def test_reduce_resolution():
    """Testing implementation of REDUCING the resolution of the DNACipher predictions, so that we do a mean pooling
        across adjacent sequence embedding positions, in order to reduce computation time!
    """

    from dnacipher_train.infer import run_effect_inference as infer

    import time
    import numpy as np

    import matplotlib.pyplot as plt

    run_name = "REDUCE_RES"

    data_dir = "/Users/bradbalderson/Desktop/projects/MRFF/data/seqcipher/revision/gtex_bench/"
    out_dir = f"{data_dir}infer_out/"

    vcf_path = f"{data_dir}gtex_highest_pip_vars.dnac_queries.txt"

    celltypes = f"{data_dir}gtex_query_tissues.txt"
    assays = f"{data_dir}gtex_query_assays.txt"

    fasta_file_path = "/Users/bradbalderson/Desktop/projects/myPython/Sheriff/example_data/hg38.fa"

    # For specifying the DNACipher model.
    fold='0'
    sample_file = f"{test_borzoi_dir}encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_{fold}.tsv"
    model_config_file = f"/Users/bradbalderson/Desktop/projects/myPython/DNACipher_Training/dnacipher_train/configs/softplusLR_model_config.yaml"
    weights_path = f"/Users/bradbalderson/Desktop/projects/MRFF/data/seqcipher/revision/borzoi/BORZOI_SOFTPLUS_LR_{fold}_model_weights.pth"

    # For specifying the embedding model.
    model_name = "borzoi"

    # General
    device = "cpu"

    index_base = 1

    correct_ref = False

    seq_pos_col = "SEQ_POS"
    effect_region_start_col = "START_EFFECT"
    effect_region_end_col = "END_EFFECT"
    batch_size = None
    batch_by = None

    all_combinations = True

    scoring_method = "sum_logfc"

    verbose = True

    infer.infer_multivariant_effects(f"{out_dir}{run_name}", vcf_path, celltypes, assays, fasta_file_path,
                                     sample_file, model_config_file, weights_path, model_name,
                                     device, index_base,
                                     correct_ref, seq_pos_col, effect_region_start_col, effect_region_end_col,
                                     batch_size,
                                     batch_by, all_combinations, scoring_method, verbose,
                                     n_vars=1, res_reduce_factor=None,
                                     )

    resolution = 4 # Reduces the embedding sequence length by a factor of 4! if DNAC-BZ (32bp) goes to equivalent of DNAC-ENF (128bp).

    res_reduce_factors = [4, 8, 12, 16]
    times = np.zeros((len(res_reduce_factors)))
    for i, res_reduce_factor in enumerate( res_reduce_factors ):
        start = time.time()
        infer.infer_multivariant_effects(f"{out_dir}{run_name}", vcf_path, celltypes, assays, fasta_file_path,
                                     sample_file, model_config_file, weights_path, model_name,
                                     device, index_base,
                                     correct_ref, seq_pos_col, effect_region_start_col, effect_region_end_col,
                                     batch_size,
                                     batch_by, all_combinations, scoring_method, verbose,
                                     n_vars=1, res_reduce_factor=res_reduce_factor,
                                     )
        elapsed = round((time.time()-start)/60, 3)
        print(f"res_reduce_factor={res_reduce_factor} in {elapsed} minutes")
        times[i] = elapsed

    plt.scatter(res_reduce_factors, times)
    plt.xlabel("Resolution reduce factor")
    plt.ylabel("Time in minutes")
    plt.show()

if __name__ == "__main__":

    ### Run different tests here.
    #test_embeds()
    #compare_embeds()
    #test_signals()
    #test_norm()
    #test_train()
    #test_eval()
    #test_multivar_inference()
    #compare_borzoi_embeds()
    #test_train()
    #trait_impact_map()
    # Working on a caching strategy so no querying by
    #test_var_effect()
    #test_new_eval()
    #test_ipscore_effect()
    #test_infer_signals()
    #test_new_loss()
    #test_eval_new_loss()
    #test_infer_signals_attn()
    #test_multivar_inference_softplus() # Only issues seemed to be the fasta file, the loaded DNACipher model looked correct.
    #investigate_strand_reversal()
    test_reduce_resolution()

