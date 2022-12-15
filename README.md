# What are you really editing?

This repository includes code for the paper [What Are You Really Editing? Surprising Differences in Where Knowledge Is Stored vs. Can Be Injected in Language Models](link). It is built on top of code from the MEMIT repository [here](https://github.com/kmeng01/memit).

## Table of Contents
1. [Installation](#installation)
2. [Causal Tracing](#causal-tracing)
3. [Model Editing](#model-editing-evaluation)
4. [Data Analysis](#data-analysis)

## Installation

For needed packages, run:
```
./scripts/setup_conda.sh  
pip install -r requirements.txt  
python -c "import nltk; nltk.download(punkt)"
```

## Causal Tracing

We gather causal tracing results from the first 2000 points in the CounterFact dataset, filtering to 652 correctly completed prompts when using GPT-J. The `window_sizes` argument controls which tracing window sizes to use. To reproduce all GPT-J results in the paper, run tracing experiments with for window sizes 10, 5, 3, and 1. This can be done with the following command:

```
python -m experiments.tracing \
    -n 2000 \
    --ds_name counterfact \
    --model_name EleutherAI/gpt-j-6B \
    --run 1 \
    --window_sizes "10 5 3 1"
```

## Model Editing Evaluation

We check the relationship between causal tracing localization and editing performance using several editing methods applied to five different variants of the basic model editing problem. The editing methods are:
- Constrained finetuning with Adam at one layer
- Constrained finetuning with Adam at five adjacent layers
- ROME (which edits one layer)
- MEMIT (which edits five layers)

The editing problems include the original model editing problem specified by the CounterFact dataset (changing the prediction for a given input), as well as a few variants mentioned below. 

```
python -m experiments.evaluate \
    -n 2000 \
    --alg_name ROME \
    --window_sizes "1" \ 
    --ds_name cf \
    --model_name EleutherAI/gpt-j-6B \
    --run 1 \
    --edit_layer -2 \
    --correctness_filter 1 \ 
    --norm_constraint 1e-4 \ 
    --kl_factor .0625
```

Add the following flags for each variation of the experiments:

- Error Injection: no flag
- Tracing Reversal: `--tracing_reversal`
- Fact Erasure: `--tracing_reversal`
- Fact Amplification: `--tracing_reversal`
- Tracing Reversal: `--tracing_reversal`

## Data Analysis

Data analysis for this work is done in R via the `data_analysis.ipynb` file. All plots and regression analyses in the paper can be reproduced via this file.

## How to Cite

```bibtex
@article{hase2023locate,
  title={Locate Then Edit? Surprising Differences in Where Knowledge Is Stored vs. Can Be Manipulated in Language Models},
  author={Peter Hase and Mohit Bansal and Been Kim and Asma Ghandeharioun},
  journal={arXiv preprint arXiv:},
  year={2023}
}
```

## Disclaimer
This is not an officially supported Google product.
