DNACipher
==================
## Prediction of genomic measurements from DNA sequence in observed and unobserved cell types and assays
<img src="https://github.com/BradBalderson/DNACipher/blob/main/img/dnacipher_logo.png" alt="DNACipher Example" width="600">

**DNACipher is a DNA sequence deep learning model that also includes cell type and assay information on the model input.**

 Tutorials - Python Interface
-------
### The following tutorials are written in google collab, showing how to use the DNACipher Python interface.

* Tutorial 1: DNACipher inference of genetic variant effects
https://colab.research.google.com/gist/BradBalderson/c4389baa0d789314259b8479cfd35747/dnacipher_inference_local.ipynb
<img src="https://github.com/BradBalderson/DNACipher/blob/main/img/dnacipher_tutorial1_figure.png" alt="DNACipher Tut1" width="1000">

* Tutorial 2: DeepVariantImpactMapping (DVIM) with DNACipher to infer common and rare genetic variants with significant effects at GWAS loci:
https://colab.research.google.com/drive/17GiWLt_SigpVa6hl6A9yP_edM4IcQeEy?usp=sharing
<img src="https://github.com/BradBalderson/DNACipher/blob/main/img/dnacipher_DVIM_example.png" alt="DNACipher Tut2" width="1000">

 Tutorials - Command-line Interface
-------
### DNACipher DVIM command-line-interface

DNACipher and DVIM analysis using a command-line interface, so that R
users and non-Python programmers can utilize the model and analysis.

***Just inferring the effects for a single variant across the given effect region***
out_prefix="/home/jovyan/data4/bbalderson_runAI/MRFF/data/dnacipher_test/WRN_eQTL_"
fasta_path="/home/jovyan/data4/bbalderson_runAI/MRFF/data/enformer/genome.fa"

chr_="chr8"
pos="31119876"
index_base="1"
ref="T"
alt="C"
seq_pos="31076838"
effect_start="31119776"
effect_end="31119976"

echo "prostate gland" > celltypes.txt
echo "plus strand polyA plus RNA-seq,minus strand polyA plus RNA-seq" > assays.txt

python -m dnacipher.main infer-effects ${chr_} ${pos} ${ref} ${alt} celltypes.txt assays.txt ${fasta_path} ${out_prefix} -i ${index_base} -s ${seq_pos} -ers ${effect_start} -ere ${effect_end}

***Inferring effects for a single variant with signals along sequence outputted***
python -m dnacipher.main infer-effects ${chr_} ${pos} ${ref} ${alt} celltypes.txt assays.txt ${fasta_path} ${out_prefix} -i ${index_base} -s ${seq_pos} -ers ${effect_start} -ere ${effect_end} -return_all

***Inferring effects for a single variant with signals along sequence outputted***
out_prefix="/home/jovyan/data4/bbalderson_runAI/MRFF/data/dnacipher_test/eQTLs_"
vcf_path="/home/jovyan/data4/bbalderson_runAI/myPython/DNACipher/tutorials/data/gtex_variants_SMALL.vcf"

python -m dnacipher.main infer-multivariant-effects ${vcf_path} celltypes.txt assays.txt ${fasta_path} ${out_prefix} -i ${index_base}

***Deep Variant Impact Mapping***
out_prefix="/home/jovyan/data4/bbalderson_runAI/MRFF/data/dnacipher_test/dvim_runx3_"
runx3_gwas_stats_path="/home/jovyan/data4/bbalderson_runAI/myPython/DNACipher/tutorials/data/Chiou-2021-T1D-GWAS_RUNX3-signal_variant_stats.txt.gz"

python -m dnacipher.main stratify-variants ${runx3_gwas_stats_path} other_allele effect_allele base_pair_location p_value effect_allele_frequency ${out_prefix}




    


