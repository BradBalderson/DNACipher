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

        mamba create dnat python=3.10

        conda activate dnat

        mamba install matplotlib seaborn pandas scipy pytorch-lightning zlib ipykernel bioconda::bedtools bioconda:pybigwig

        export LDFLAGS="-L$CONDA_PREFIX/lib"
        export CPPFLAGS="-I$CONDA_PREFIX/include"
        export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

        pip install numba typer pyfaidx kipoiseq enformer-pytorch pybedtools pyBigWig borzoi-pytorch

        git clone https://github.com/BradBalderson/DNACipher
        cd DNACipher
        pip install .

Verify install with:
```bash
dnacipher version
```
    DNACipher 1.5.0: 🦾MetalGrip

Depending on your system, the above may not install the correct Pytorch to detect your GPU.
To check this, run the following:
```bash
dnacipher device
```
    Will use cuda GPU

<details>
<summary><strong>Trouble shooting no GPU detected:</strong></summary>
If you have a GPU but it was not detected, please check you have installed the correct Pytorch by:

```bash
nvidia-smi
```

Then cross-reference with this Pytorch resource to ensure you are installing the correct Pytorch:

https://pytorch.org/get-started/locally/ 

Once you have done this, re-check if the GPU is detected with the ***dnacipher device*** command.

</details>

You will need to figure out additional installs if using a new sequence embedding model, but the above does the 
appropriate install for Enformer and Borzoi.

There are also additional optional installs that enable faster precomputation of the signal matrices across experiments 
that are used for training, and is detailed in the data setup section for adding additional data, because is otherwise
unnecessary because I provide these precomputed.

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

    PRECOMP-MID-AVG_GENOME_genome_rois.train_test.borzoi.txt -> The train/test genome regions, that have been compiled 
                                                                            with the Borzoi train/test regions in-mind.

    PRECOMP-MID-AVG_GENOME_genome_rois.train_test.borzoi.100N.split.txt -> Additionally indexes where the signals / 
                                           embeddings for these regions occur in the '.split.h5', which is a file format
                                           that stores sub-matrices of the data within the .h5, and the index helps with 
                                            a fast lookup, and enables faster reading
                                            of the data during train time.

    REPROD_BWTOOL_epivalues.100N.split.h5 -> Contains the precomputed average signal values at each of the train/test
                                        regions across all of the experiments in the peak_bigwigs folder. 
                                        As mentioned, is in the .split.h5 format.

    REPROD_EMBED_enformer_embeds.100N.split.h5 -> Precomputed Enformer embeddings for each of the train/test regions.

    BORZOI_EMBED_borzoi_embeds.100N.split.h5 -> Precomputed Borzoi embeddings for each of the train/test regions.

The other important files is the metadata associated with each bigwig file, which are specified in:

    encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_*.tsv 

Additional knowledge of this file is only necessary if training with additional data, detailed in the next section.

<details>
<summary><strong>OPTIONAL: Including additional data</strong></summary>

The signal values are already precomputed as specified in the data repository above, but if you want to include
additional data the following steps below are necessary.

An important file in the data repository if wanting to include a new dataset for DNACipher training is the 
peak_bigwigs.tar.gz, which provides bigWig signal files for each experiment. 
How these bigWig files were generated, as described in the manuscript, is shown in the notebooks included in the data
respository.

This needs to be decompressed:

    tar -xzvf peak_bigwigs.tar.gz

Which will generate 2,882 .bigWig files within a folder peak_bigwigs.

Additionally, to train on new data, you will need to follow the appropriate ENCODE preprocessing pipeline for the 
assay type: https://www.encodeproject.org/pipelines/

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

Most of these fields are ENCODE metadata. For new data, the fields you NEED to specify are:

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
to train/test overall performances, and therefore for your additional datasets, the specific performances can be observed
from these detailed outputs.

Once you have the following you have performed all necessary steps to continue to the next step of training with your additional dataset:

    1 added the additional dataset metadata to a sample metadata file (eg encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_0.tsv) 
    2 appropriate .bigWig file in peak_bigwigs

***PRECOMPUTING SIGNAL MATRICES FOR TRAINING***

Now we need to precompute the .h5 signal files from the collection of bigwigs within peak_bigwigs/. 
Note that this step is already computed for the existing data, so you don't need to do it unless you have additional data for training.

By default, you can use pyBigWig already installed for this precomputation step, but it is VERY slow
compared to bwtool, albeit easier to install. So if you find signal generation too slow for you, you should follow the 
installation steps below.

<details>
<summary><strong>OPTIONAL: Faster signal precomputation with bwtool</strong></summary>

bwtool has greater performance but does require additional install steps that were non-trivial for me to determine, 
so will include exact details on the debugging process I went through also to help others debug their particular machine.
These steps were for a linux/ubuntu OS.

See: https://github.com/CRG-Barcelona/bwtool/wiki

Install instructions provided on the wiki don't work. 

But these do from the github issues: https://github.com/CRG-Barcelona/bwtool/issues/49

Particularly comments from PoisonAlien for global install, and hoondy for local. Following local below.

        mkdir /home/<you_username>/local
        mkdir /home/<you_username>/local/include
        mkdir /home/<you_username>/local/lib

        git clone https://github.com/madler/zlib
        cd zlib
        ./configure --prefix=$HOME
        make
        cd ..

        git clone https://github.com/CRG-Barcelona/libbeato.git
        cd libbeato/
        git checkout 0c30432af9c7e1e09ba065ad3b2bc042baa54dc2
        ./configure --prefix=$HOME
        make
        cd ..

        git clone https://github.com/CRG-Barcelona/bwtool.git
        cd bwtool/
        ./configure --prefix=$HOME CFLAGS='-I../libbeato -I../zlib' LDFLAGS='-L../libbeato/jkweb -L../libbeato/beato -L../zlib'
        make
        make install

You should now have a bwtool binary that can very quickly get average signals for pre-specified regions from a bigWig input,
and the DNACipher CLI when pointed to this binary will use it to precompute signals and then re-format them to .h5.

</details>

Now you can compile all of the signals from across the 2,882 + additional data .bigwig files, using the DNACipher CLI.
Below I include the commands that I used for the DNACipher manuscript, and the output files from this are provided
in the data repository detailed above. If you did not install bwtool above, you can use 'dataloader' instead for the 
'method' command below, and this will use pyBigWig in the backend. I also implemented using 'deeptools', but this actually
uses pyBigWig in the backend also so gave no speed improvements, so I will not bother including install instructions.

This step took ~3 hrs when I ran it with bwtool as the backend. It will tens of hours to days with the other backends.

    TEST_NAME="REPROD_BWTOOL"
    GENOME_FILE="PRECOMP-MID-AVG_GENOME_genome_rois.train_test.borzoi.txt"
    BIGWIG_DIR="peak_bigwigs/"
    SAMPLE_FILE="encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_0.tsv"
    BATCH_SIZE="100"
    CPU="20"
    REGION_PAD="147"
    OUT_DIR="."
    method="bwtool"
    
    dnacipher precompute-signals $TEST_NAME $GENOME_FILE $BIGWIG_DIR $SAMPLE_FILE -b $BATCH_SIZE -c $CPU -r $REGION_PAD -o $OUT_DIR -m $METHOD

This command will output:

    ${OUT_DIR}REPROD_BWTOOL_epivalues.h5

Internally, the .h5 stores a matrix of size N_REGIONS X N_EXPERIMENTS, with the signal values for each region in each
experiment. This is quite big, N_REGIONS > 1 million, N_EXPERIMENTS > 2,882 with additional data.

It can be slow to load with standard .h5 loading, but I developed this .split.h5 format, that splits the rows of the 
matrix into separate matrices and stores them, and makes for much faster reading, albeit it requires an index to store
where the index in the larger matrix falls into which split matrix and it's corresponding row. We will generate this
file now, it runs very quickly:

    TEST_NAME="FULL_SPLIT"
    GENOME_FILE="PRECOMP-MID-AVG_GENOME_genome_rois.train_test.borzoi.txt"
    H5="${OUT_DIR}REPROD_BWTOOL_epivalues.h5"
    N="100"
    dnacipher split-h5 $TEST_NAME $GENOME_FILE $H5 $N

The N specifies how many sub-matrices to create when splitting up the rows of the matrix; N=100 means we split
the >1.3million rows into 100 separate matrices and store in a new .split_h5, and the details of where each region
in these sub-matrices is stored is detailed in the newly outputed GENOME_FILE which is the index lookup. So we have 
these outputs:

    REPROD_BWTOOL_epivalues.100N.split.h5
    PRECOMP-MID-AVG_GENOME_genome_rois.train_test.borzoi.100N.split.txt

The DNACipher CLI can read these file formats. These outputs for the full dataset are provided in the dataset repository above.
</details>

<details>
<summary><strong>OPTIONAL: (Re)Generating sequence embeddings from (new)models</strong></summary>

The sequence embeddings for the train/test regions for Enformer and Borzoi are already precomputed and specified in the 
data repository above, but if you want to recompute these for some reason, or compute the sequence embeddings from a 
new model for the train regions, or perhaps compute for completely different regions, the following steps are necessary.

Note that if you change the regions for some reason, you must also recompute the signals as detailed in the section
***Including additional data***, where you will precompute the signal values for these new regions so the embeddings and
the signals are aligned and refer to the same genome regions.

***IMPORTANT CONSIDERATION IF ADDING SEQUENCE MODEL ALSO TRAINED FROM SIGNAL DATA***
If using a sequence embedding model that was trained on signal data previously, you must consider the train/test
sequence allocation of that model, to prevent data leakage for the train/test sequences for the DNACipher training. To
do this, simply update the PRECOMP-MID-AVG_GENOME_genome_rois.train_test.borzoi.txt which specifies the train/test
regions, by re-labelling the existing regions as train/test as appropriate to the underlying sequence embedding model. 
Furthermore, this will also necessitate changing the train/test experiment allocation in the file 
encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_0.tsv to ensure the 'test' experiments
are not experiments that the underlying sequence model has been trained on. This is as simple as updating the 'allocation'
field in the sample metadata, which will also require updating the normalisation of the underlying signals (see below).
A limitation of this approach is if the sequence embedding model was trained on all prior experiments currently supported,
in this case, you must include additional experiments to verify the imputation is generalising 
(see ***Including additional data***) or re-train the underlying sequence model with less data, or allocate some experiments 
to test but keep in-mind their performance metrics for signal prediction may be inflated as they are not truly unobserved
experiments because of pre-training leakage from the sequence embedding model pre-training.

<details>
<summary><strong>OPTIONAL: Specifying a new sequence embedding model</strong></summary>

Because the DNACipher CLI needs to be able to generate sequence embeddings from the underlying model on-the-fly when 
queried, it is necessary to implement a wrapper for the sequence embedding model within DNACipher so this can be done
in a consistent way to interface with the rest of the DNACipher code.

Therefore, you need to add an additional class to DNACipher that is an instance of the 'EmbeddingModel' class specified
in the following script:

    DNACipher/dnacipher/embed/embedding_model.py

An example of this for Enformer and Borzoi can be seen here:

    DNACipher/dnacipher/embed/borzoi_embed.py
    DNACipher/dnacipher/embed/enformer_embed.py

Critically, you must implement functions that instantiate the underlying sequence model and attach it to the class
with the appropriate parameters, with the critical parameters that DNACipher needs for embedding generation specified 
as the inputs for the EmbeddingModel class (see the doc-string there for details on these parameters). Additionally,
you must implement a 'get_sequence_embeddings' function, which takes a list of DNA sequence strings of length specified
by the model, and outputs sequence embeddings as also specified when creating the EmbeddingModel class. These parameters
are important so the DNACipher EmbeddingGenerator (see DNACipher/dnacipher/embed/embedding_generator.py) knows
what to expect from the embedding model outputs.

Once you have implemented your EmbeddingModel class, import it and add it to DNACipher/dnacipher/embed/md.py,
as shown for Enformer and Borzoi. This is where DNACipher will look for supported embedding models. 

Finally, once this is complete, you must now re-install the DNACipher CLI so that when you call it from the commandline
you are working with the version that supports your newly implemented EmbeddingModel:

    cd DNACipher
    pip install .

If you managed to appropriately implement a new embedding model, please feel free to put in a pull request and I will
review it before merging into the main DNACipher repository. We could also coordinate hosting the precomputed
embeddings on the data repository for easier training for new datasets. See contact details in README.md. Thanks!
</details>

Precomputing embeddings at the appropriate sequence regions can be done via the commandline interface as shown below.
To change the model used, change the 'MODEL' parameter. Currently 'enformer' and 'borzoi' are implemented.

If you added a new sequence embedding model in the above optional step, specify the name of the model you implemented,
which is specified by the 'model_name' parameter on the instance of 'EmbeddingModel' that you implemented.

If you want to change the regions, you need to change 'GENOME_FILE'.

The below parameters are the exact same used for the enformer embeddings used to train the DNACipher-Enformer model,
and if you change MODEL to "borzoi" it is the exact same embeddings used to train the DNACipher-Borzoi model.

This step took 2.1 hrs when I ran it for Enformer.

    TEST_NAME="REPROD_EMBED"
    GENOME_FILE="PRECOMP-MID-AVG_GENOME_genome_rois.train_test.borzoi.txt"
    FASTA_FILE="hg38.fa"
    MODEL="enformer"
    GPU="cuda:0"
    BATCH_SIZE="448"
    REGION_PAD="147"
    OUT_DIR="."
    dnacipher precompute-embeds ${TEST_NAME} ${GENOME_FILE} ${FASTA_FILE} ${MODEL} ${GPU} -b ${BATCH_SIZE} -r ${REGION_PAD} -o ${OUT_DIR}

This command outputs:

    REPROD_EMBED_enformer_embeds.h5

We now convert this to a .split.h5. CRITICALLY, you must specify the same N as for precomputing the signal values for this
step, so that the index generated can be used for both files when the DNACipher CLI pulls the region values and the region
sequence embedding from these files. This step should be extremely fast, <5 min.

    TEST_NAME="EMBED_SPLIT"
    GENOME_FILE="PRECOMP-MID-AVG_GENOME_genome_rois.train_test.borzoi.txt"
    H5="${OUT_DIR}REPROD_EMBED_enformer_embeds.h5"
    N="100"
    dnacipher split-h5 $TEST_NAME $GENOME_FILE $H5 $N

This outputs:

    REPROD_EMBED_enformer_embeds.100N.split.h5
    PRECOMP-MID-AVG_GENOME_genome_rois.train_test.borzoi.100N.split.txt

Which is provided in the data repository. The equivalent for Borzoi embeddings is also provided, to make the training
steps easier. 

</details>


### Normalisation of the precomputed signals.

This is an essential step, and it is performed separately for each cross-fold validation, because normalisation can be
a source of data leakage, and therefore the normalisation parameters should be fit to the train signals,
then applied separately to the test signals. 

It very fast to perform, and the required files are provided in the data repository. If you performed an of the optional
steps above, you will need to update the below commands appropriately.

    TEST_NAME="BORZOI_NORM_0"
    GENOME_FILE="PRECOMP-MID-AVG_GENOME_genome_rois.train_test.borzoi.100N.split.txt"
    SAMPLE_FILE="encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_0.tsv"
    H5_FILE="REPROD_BWTOOL_epivalues.100N.split.h5"
    dnacipher norm $TEST_NAME $GENOME_FILE $SAMPLE_FILE $H5

I ran this separately for each of the encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_*.tsv
files, so there were separate normalisations for each fold of train/test experiments. Note that this is all applied 
internally, so there is not a separate output for train/test experiments, but internally the normalisation is fit to the
train signals then applied to the test signals, as specified by the input files.

This command will output:

    REPROD_BWTOOL_epivalues.BORZOI_NORM_0.normalised.100N.split.h5
    BORZOI_NORM_0_REPROD_BWTOOL_epivalues._NORMED_epimeasure_distribs.png

The .png is a visualisation of the non-zero signals per assay type, and should have approximately the same mean and a
relatively similar data range for each assay type. If this is not the case, something has gone wrong, and the data
is not appropriately normalised which could effect downstream training. When incorporating new assay types, this could
occur and is an important check to make, and you may need bespoke normalisation to fix this, similar to the additional
steps in the normalisation I described for the RNA-seq data.

 2.0 TRAIN
-------
With the completion of prior steps, we now have all the data and files necessary to train a DNACipher model.

The training commands below generate a *log_file.txt that includes performance metrics and other run-time details and 
output plots, but the key output is:

    BORZOI_SOFTPLUS_LR_0_model_weights.pth
    ENFORMER_SOFTPLUS_LR_0_model_weights.pth

These are model weights, and these are provided for when I ran the commands below in:

    DNACipher/dnacipher/weights/

They are read internally by the DNACipher CLI when performing variant effect inference when specifying either model.

Below is the exact commands used to train DNACipher-Borzoi and DNACipher-Enformer used throughout the manuscript 
(the 5-fold cross validation train the same, just changed the 'FOLD' parameter). 

    PREFIX="BORZOI_SOFTPLUS_LR"

    GENOME_FILE="PRECOMP-MID-AVG_GENOME_genome_rois.train_test.borzoi.100N.split.txt"
    EMBED_FILE="BORZOI_EMBED_borzoi_embeds.100N.split.h5"
    OUT_DIR="."
    BATCH_SIZE="256"
    CPU="20"
    EPOCHS_WARMUP="25"
    EPOCHS_FINAL="25" 
    EVAL_BATCHES="50"
    LEARNING_RATE="0.0014526"
    N_REGIONS="999999999999"
    CONFIG="DNACipher_Training/dnacipher_train/configs/softplusLR_model_config.yaml"

    FOLD="0"

    TRAIN_NAME="${PREFIX}_${F}"

    SAMPLE_FILE="encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_${F}.tsv"
    SIGNAL_FILE="REPROD_BWTOOL_epivalues.BORZOI_NORM_${F}.normalised.100N.split.h5"

	dnacipher_train train $TRAIN_NAME $GENOME_FILE $SAMPLE_FILE $SIGNAL_FILE $EMBED_FILE -o $OUT_DIR -b $BATCH_SIZE \
	-c $CPU -ew $EPOCHS_WARMUP -ef $EPOCHS_FINAL -eb $EVAL_BATCHES -lr $LEARNING_RATE -nr ${N_REGIONS} \
    -m ${CONFIG}

And additionally for DNACipher-Enformer:

    PREFIX="ENFORMER_SOFTPLUS_LR"

    GENOME_FILE="PRECOMP-MID-AVG_GENOME_genome_rois.train_test.borzoi.100N.split.txt"
    EMBED_FILE="REPROD_EMBED_enformer_embeds.100N.split.h5"
    OUT_DIR="."
    BATCH_SIZE="256"
    CPU="20"
    EPOCHS_WARMUP="25"
    EPOCHS_FINAL="25" 
    EVAL_BATCHES="50"
    LEARNING_RATE="0.0014526"
    N_REGIONS="999999999999"
    CONFIG="DNACipher_Training/dnacipher_train/configs/softplusLR_model_config.yaml"

    FOLD="0"

    TRAIN_NAME="${PREFIX}_${F}"

    SAMPLE_FILE="encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_${F}.tsv"
    SIGNAL_FILE="REPROD_BWTOOL_epivalues.BORZOI_NORM_${F}.normalised.100N.split.h5"

	dnacipher_train train $TRAIN_NAME $GENOME_FILE $SAMPLE_FILE $SIGNAL_FILE $EMBED_FILE -o $OUT_DIR -b $BATCH_SIZE \
	-c $CPU -ew $EPOCHS_WARMUP -ef $EPOCHS_FINAL -eb $EVAL_BATCHES -lr $LEARNING_RATE -nr ${N_REGIONS} \
    -m ${CONFIG}

Note how it is the EXACT same command, except we just changed the embedding file and run name. The DNACipher CLI will detect
different embedding dimensions for the sequences, and adjust the resulting model architecture outputted appropriately.

For further DNACipher model adjustments, such as the size of the cell type and assay embeddings, you can copy and adjust:

    DNACipher_Training/dnacipher_train/configs/softplusLR_model_config.yaml

NOTE that in the first preprint version, I used 'default_model_config.yaml', which used the ReLU transformation on model 
output for 'EPOCHS_FINAL'. But the new version uses SOFTPLUS throughout, but the EPOCHS_FINAL does additionally perform
weighted MSE depending on the rarity of each experiment in the training set (see manuscript Methods for details, or
read the code directly).

#### Inference with new model weights

Performing variant effect inference with DNACipher is detailed in the README.md. But with a newly trained model, you 
will need to change the model weights, and specify the appropriate embedding model. 

In the example below, if you did not alter the data but did train a new model, you would re-specify:

    WEIGHTS_PATH
    MODEL_NAME

If you changed the architecture by adjusting the config file, you would also need to specify:
    
    MODEL_CONFIG_FILE

If you change the data, example adding additional cell types or assays, you would need to specify:

    SAMPLE_FILE

Here is the inference example for adjusting the CLI to use the Borzoi weights:

    TEST_NAME="DNAC_BZ_GTEX"
    
    OUT_PREFIX="${TEST_NAME}"
    QUERY_PATH="gtex_highest_pip_vars.dnac_queries.txt"
    CELLTYPES="gtex_query_tissues.txt"
    ASSAYS="gtex_query_assays.txt"
    FASTA_FILE_PATH="hg38.fa"
    
    SAMPLE_FILE="encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_0.tsv"
    MODEL_CONFIG_FILE="DNACipher_Training/dnacipher_train/configs/softplusLR_model_config.yaml"
    WEIGHTS_PATH="BORZOI_SOFTPLUS_LR_0_model_weights.pth"
    MODEL_NAME="borzoi"
    DEVICE="cuda:0"
    INDEX_BASE="1"
    CORRECT_REF="False"
    SEQ_POS_COL="SEQ_POS"
    EFFECT_REGION_START_COL="START_EFFECT"
    EFFECT_REGION_END_COL="END_EFFECT"
    BATCH_SIZE="424"
    BATCH_BY="None"
    ALL_COMBINATIONS="True"
    SCORING_METHOD="sum_logfc"
    VERBOSE="True"
    
    CPU="10"

    dnacipher infer-multivariant-effects \
            $OUT_PREFIX \
            $QUERY_PATH \
            $CELLTYPES \
            $ASSAYS \
            $FASTA_FILE_PATH \
            $( [ "$SAMPLE_FILE" != "None" ] && echo "-sf $SAMPLE_FILE" ) \
            $( [ "$MODEL_CONFIG_FILE" != "None" ] && echo "-m $MODEL_CONFIG_FILE" ) \
            $( [ "$WEIGHTS_PATH" != "None" ] && echo "-w $WEIGHTS_PATH" ) \
            $( [ "$MODEL_NAME" != "None" ] && echo "-mn $MODEL_NAME" ) \
            $( [ "$DEVICE" != "None" ] && echo "-d $DEVICE" ) \
            $( [ "$INDEX_BASE" != "None" ] && echo "-i $INDEX_BASE" ) \
            $( [ "$CORRECT_REF" = "True" ] && echo "-correct_ref" || echo "-no-correct_ref" ) \
            $( [ "$SEQ_POS_COL" != "None" ] && echo "-sc $SEQ_POS_COL" ) \
            $( [ "$EFFECT_REGION_START_COL" != "None" ] && echo "-ersc $EFFECT_REGION_START_COL" ) \
            $( [ "$EFFECT_REGION_END_COL" != "None" ] && echo "-erec $EFFECT_REGION_END_COL" ) \
            $( [ "$BATCH_SIZE" != "None" ] && echo "-b $BATCH_SIZE" ) \
            $( [ "$BATCH_BY" != "None" ] && echo "-by $BATCH_BY" ) \
            $( [ "$ALL_COMBINATIONS" = "True" ] && echo "-all_combinations" || echo "-no-all_combinations" ) \
            -sm $SCORING_METHOD \
            $( [ "$VERBOSE" = "True" ] && echo "-verbose" || echo "-quiet" 

<details>
<summary><strong>OPTIONAL: EXTENSIVE EVALUATION</strong></summary>

The default training command above will use 50 batches of 256 train/test sequences across the train/test experiments to
do evaluation for measuring signal prediction correlations for each stratification of train/test sequences and train/test
experiments. It does not however do all of the data, and it does not compare against the mean kNN baseline as described
in the manuscript. The below command does this, by specifying more evaluation batches (EVAL_BATCHES) and including the 
precomputed kNN cell types (celltype_knn.json) and might be something to do for any new models to ensure a rigorous 
evaluation:

    GENOME_FILE="PRECOMP-MID-AVG_GENOME_genome_rois.train_test.borzoi.100N.split.txt"
    EMBED_FILE="REPROD_EMBED_enformer_embeds.100N.split.h5"
    OUT_DIR="."
    BATCH_SIZE="256"
    CPU="20"
    EVAL_BATCHES="1350"
    N_REGIONS="999999999999"
    CONFIG="/home/uqbbalde/myPython/DNACipher_Training/dnacipher_train/configs/softplusLR_model_config.yaml"

    DEVICE="cuda:0"
    CT_KNN="DNACipher/dnacipher/configs/celltype_knn.json"

    TRAIN_PREFIX="ENFORMER_SOFTPLUS_LR"
    PREFIX="EVAL_${TRAIN_PREFIX}"

    FOLD="0"
    EVAL_NAME="${PREFIX}_${F}"
    WEIGHTS_PATH="${TRAIN_PREFIX}_${F}_model_weights.pth"
    SAMPLE_FILE="encode_meta_encode_imputable_filter-fixed_train-test_samp-probs.borzoi_fold_${F}.tsv"
    SIGNAL_FILE="REPROD_BWTOOL_epivalues.BORZOI_NORM_${F}.normalised.100N.split.h5"

    dnacipher eval $EVAL_NAME $GENOME_FILE $SAMPLE_FILE $SIGNAL_FILE $EMBED_FILE $WEIGHTS_PATH -o $OUT_DIR -b $BATCH_SIZE -c $CPU -d $DEVICE -eb $EVAL_BATCHES -ct_knn ${CT_KNN} -m ${CONFIG}

There will be alot of outputs from this command running many evals, including *kNN* stats files, which includes the 
batch correlation tests between the DNACipher predictions and the kNN baseline, which was used to determine the 
proportion of assays predicted where DNACipher outperform the kNN baseline. This will likely be a useful metric to 
compare for any new DNACipher models you train, in addition to the overall Pearson's correlations in the *log_file.txt,
and the per-experiment correlations that are also outputted.

</details>

 3.0 Post-training
-------

After training a model, you can now use the models weights freely for variant effect inference for the molecular contexts
represented, including the new contexts you may have provided with the instructions above, or improved predictions with
the new sequence embedding model you may have included. If combined with DVIM, perhaps your new DNACipher model may yield 
improved imVar detection at GWAS loci, and generated additional trait-mediating molecular effect hypotheses. 
See the README.md on how to do this with the newly trained model.

If you train a new model and use it in your work, please cite below, and feel free to email me any questions and please
let me know if you train new models; we can work on incorporating them into the CLI so other can easily use.

Citation
--------

***NOTE the below preprint to be updated soon***

***Comprehensive molecular impact mapping of common and rare variants at GWAS loci.***
Brad Balderson, Sanjana Tule, Mei-Lin Okino, William JF Rieger, Sierra Corban, Jeff Jaureguy, Nathan Palpant, Kyle J. Gaulton, Mikael Boden, Graham McVicker
bioRxiv 2025.06.05.658079; doi: https://doi.org/10.1101/2025.06.05.658079

Contact
-------

Authors: Brad Balderson

Contact:  uqbbalde@uq.edu.au

    

    


