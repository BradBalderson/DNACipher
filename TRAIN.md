🦾DNACipher Training
==================

NOTE for adding additional data or updating to a new underlying sequence embedding model, this is for advanced users,
since careful consideration needs to be made to prevent train/test data leakage and appropriate dataset preprocessing
prior to model training, in addition to being able to appropriately deploy training to a GPU.

 0.0 Installation
-------
Training requires additional installation compared to just running inference. The following installation steps
will enable all downstream training, though additional installation will be needed for the appropriate ENCODE pipeline
if including additional data in training.



 1.0 Data Setup
-------

The hg38 reference genome was used for all experiments, so needs to be the reference genome used for training unless
wanting to re-process all files. Note that although hg38 was used for training, other reference genomes can be used
for inference if not wanting to liftOver variant locations.

Automated steps for reference genome download:

```bash
wget -O - http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz | gunzip -c > hg38.fa

mamba install bioconda::samtools
samtools faidx hg38.fa
 ```  

Preprocessed ENCODE data can be downloaded from UQ eSPACE here: https://doi.org/10.48610/c5d0e92

If training the model for reproduction purposes, I have precomputed the key data:

    PRECOMP-MID-AVG_GENOME_genome_rois.train_test.borzoi.txt -> The train/test genome regions, that have been compiled with the Borzoi train/test regions in-mind.

    PRECOMP-MID-AVG_GENOME_genome_rois.train_test.borzoi.100N.split.txt -> Additionally indexes where the signals / embeddings for these regions occur in the '.split.h5', which is a file format
    that stores sub-matrices of the data within the .h5, and the index helps with a fast lookup, and enables faster reading
    of the data during train time.

    REPROD_BWTOOL_epivalues.100N.split.h5 -> Contains the precomputed average signal values at each of the train/test regions across all of the experiments in the peak_bigwigs folder. As mentioned, is in the .split.h5 format.

    REPROD_EMBED_enformer_embeds.100N.split.h5 -> Precomputed Enformer embeddings for each of the train/test regions.

    BORZOI_EMBED_borzoi_embeds.100N.split.h5 -> Precomputed Borzoi embeddings for each of the train/test regions.

The other important files is the metadata associated with each bigwig file, which are specified in:

    encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_*.tsv 

Additional knowledge of this file is only necessary if training with additional data, detailed in the next section.

<details>
<summary><strong>OPTIONAL: Including additional data</strong></summary>

An important file in the data repository if wanting to include a new dataset for DNACipher training is the 
peak_bigwigs.tar.gz, which provides bigWig signal files for each experiment. 
How these bigWig files were generated, as described in the manuscript, is shown in the notebooks included in the data
respository.

This needs to be decompressed:

    tar -xzvf peak_bigwigs.tar.gz

Which will generate 2,882 .bigWig files within a folder peak_bigwigs.

Additionally, to train on new data, you will need to follow the appropriate ENCODE preprocessing pipeline for the 
assay type:

    https://www.encodeproject.org/pipelines/

Single cell RNA-seq data is currently not supported, but an advanced user could do it.

For sc/sn ATAC-seq, the second read pair library structure for 10x is identical bulk ATAC-seq, so you can split the reads by cell type 
if you have cell type annotations for the cell barcodes specified in read pair 1 and then follow the standard bulk ATAC-seq 
processing workflow for the separate FASTQ files provided by ENCODE to generate the appropriate output files for the
corresponding cell type.

From those output files, if it is ChIP-seq/ATAC-seq/DNAse-seq, the appropriate files to use will be peak files, which 
will then need to be converted to .bigWig files, as detailed in the data repository notebooks.

For RNA-seq, it will be the '[plus/minus] strand signal of unique reads'. The exact output type used from ENCODE pipelines
for each file is detailed in the following sample metadata files in the data repository:

    encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_*.tsv 

Each of these files are the same, except for the train/test allocation field, which specifies the 5-fold cross-validations
of train/test experiments.

When including additional data, you must add information about this data to this metadata as an additional row:

    	File accession	File format	File type	File format type	Output type	File assembly	Experiment accession	Assay	Donor(s)	Biosample term id	Biosample term name	Biosample type	Biosample organism	Biosample treatments	Biosample treatments amount	Biosample treatments duration	Biosample genetic modifications methods	Biosample genetic modifications categories	Biosample genetic modifications targets	Biosample genetic modifications gene targets	Biosample genetic modifications site coordinates	Biosample genetic modifications zygosity	Experiment target	Library made from	Library depleted in	Library extraction method	Library lysis method	Library crosslinking method	Library strand specific	Experiment date released	Project	RBNS protein concentration	Library fragmentation method	Library size range	Biological replicate(s)	Technical replicate(s)	Read length	Mapped read length	Run type	Paired end	Paired with	Index of	Derived from	Size	Lab	md5sum	dbxrefs	File download URL	Genome annotation	Platform	Controlled by	File Status	s3_uri	Azure URL	File analysis title	File analysis status	Audit WARNING	Audit NOT_COMPLIANT	Audit ERROR	Assay2	celltype_assay	allocation	sampling_weights	sampling_assayBiased
    ENCFF966BXX	bigWig	bigWig		minus strand signal of unique reads	GRCh38	ENCSR403SZN	total RNA-seq	/human-donors/ENCDO271OUW/	UBERON:0001157	transverse colon	tissue	Homo sapiens											RNA	rRNA				reverse	2016-08-04	ENCODE		see document	>200	1	1_1		/files/ENCFF636PCD/, /files/GRCh38_EBV.chrom.sizes/	30378370	ENCODE Processing Pipeline	8279484f28f25542a46d52a8d01fb404		https://www.encodeproject.org/files/ENCFF966BXX/@@download/ENCFF966BXX.bigWig	V29			released	s3://encode-public/2020/07/15/7aa876fe-bcfb-4170-bd7b-43dee4e5bae2/ENCFF966BXX.bigWig	https://datasetencode.blob.core.windows.net/dataset/2020/07/15/7aa876fe-bcfb-4170-bd7b-43dee4e5bae2/ENCFF966BXX.bigWig?sv=2019-10-10&si=prod&sr=c&sig=9qSQZo4ggrCNpybBExU8SypuUZV33igI11xw0P7rB3c%3D	ENCODE4 v1.1.0 GRCh38 V29	released				minus strand total RNA-seq	transverse colon---minus strand total RNA-seq	train	105	42

Most of these fields are ENCODE metadata. For new data, the most important fields to specify are:

    File accession, celltype_assay, allocation

'File accession' is used as a prefix to the .bigWig file, and DNACipher CLI will look in peak_bigwig/ for a file with this
prefix and '{file_accession}.bigWig'. 

celltype_assay is in format '{celltype}---{assay}' and is used to specify the cell type/assay that the experiment represents,
which the DNACipher model will one hot encode when reading the signal data to contextualise the observed signal during training.
If there is a very similar cell type/tissue specified already in the metadata, I would recommend using that label to share
information with all other experiments present for that cell type. If your cell type is sufficiently different, feel free
to label is something else, and the model will learn how it relates to the other cell type labels. This is also true for 
assay type, however, for any kind of RNA-seq or other assay using signal of unique reads, this will be normalise slightly differently 
to get it on the same scale as the other assay types represented as peak signals, so it is important to include 'RNA-seq'
in the name so the downstream normalisation will handle that appropriately.

allocation is 'train' or 'test'. ***IMPORTANT*** when doing this allocation, the underlying training data from the original
sequence embedding model must be considered to avoid inflated signal imputation metrics. Regardless of the stratification
however, the DNACipher training pipeline will output performance metrics for all experiments independently in addition
to train/test overall performances, and therefore for your additional datasets, the specific performances can be observed.

Once you have the following you have performed all necessary to continue to the next step of training with your additional dataset:

    1 added the additional dataset metadata to a sample metadata file (eg encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_0.tsv) 
    2 appropriate .bigWig file in peak_bigwigs





