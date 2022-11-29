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


--

This is not an official Google product.

<<<<<<< PATCH SET (eb48a5 staging for public release)
=======
See [`baselines/`](baselines/) for a description of the available baselines.

### Running the Full Evaluation Suite

[`experiments/evaluate.py`](experiments/evaluate.py) can be used to evaluate any method in [`baselines/`](baselines/).
To get started (e.g. using ROME on GPT-2 XL), run:
```bash
python3 -m experiments.evaluate \
    --alg_name=ROME \
    --model_name=gpt2-xl \
    --hparams_fname=gpt2-xl.json
```

Results from each run are stored at `results/<method_name>/run_<run_id>` in a specific format:
```bash
results/
|__ ROME/
    |__ run_<run_id>/
        |__ params.json
        |__ case_0.json
        |__ case_1.json
        |__ ...
        |__ case_10000.json
```

To summarize the results, you can use [`experiments/summarize.py`](experiments/summarize.py):
```bash
python3 -m experiments.summarize --dir_name=ROME --runs=run_<run_id>
```

Running `python3 -m experiments.evaluate -h` or `python3 -m experiments.summarize -h` provides details about command-line flags.

### Integrating New Editing Methods

<!-- Say you have a new method `X` and want to benchmark it on CounterFact. Here's a checklist for evaluating `X`:
- The public method that evaluates a model on each CounterFact record is [`compute_rewrite_quality`](experiments/py/eval_utils.py); see [the source code](experiments/py/eval_utils.py) for details.
- In your evaluation script, you should call `compute_rewrite_quality` once with an unedited model and once with a model that has been edited with `X`. Each time, the function returns a dictionary. -->

Say you have a new method `X` and want to benchmark it on CounterFact. To integrate `X` with our runner:
- Subclass [`HyperParams`](util/hparams.py) into `XHyperParams` and specify all hyperparameter fields. See [`ROMEHyperParameters`](rome/rome_hparams.py) for an example implementation.
- Create a hyperparameters file at `hparams/X/gpt2-xl.json` and specify some default values. See [`hparams/ROME/gpt2-xl.json`](hparams/ROME/gpt2-xl.json) for an example.
- Define a function `apply_X_to_model` which accepts several parameters and returns (i) the rewritten model and (ii) the original weight values for parameters that were edited (in the dictionary format `{weight_name: original_weight_value}`). See [`rome/rome_main.py`](rome/rome_main.py) for an example.
- Add `X` to `ALG_DICT` in [`experiments/evaluate.py`](experiments/evaluate.py) by inserting the line `"X": (XHyperParams, apply_X_to_model)`.

Finally, run the main scripts:
```bash
python3 -m experiments.evaluate \
    --alg_name=X \
    --model_name=gpt2-xl \
    --hparams_fname=gpt2-xl.json

python3 -m experiments.summarize --dir_name=X --runs=run_<run_id>
```

### Note on Cross-Platform Compatibility

We currently only support methods that edit autoregressive HuggingFace models using the PyTorch backend. We are working on a set of general-purpose methods (usable on e.g. TensorFlow and without HuggingFace) that will be released soon.

<!-- 
Each method is customizable through a set of hyperparameters. For ROME, they are defined in `rome/hparams.py`. At runtime, you must specify a configuration of hyperparams through a `.json` file located in `hparams/<method_name>`. Check out [`hparams/ROME/default.json`](hparams/ROME/default.json) for an example.

At runtime, you must specify two command-line arguments: the method name, and the filename of the hyperparameters `.json` file.
```bash
python3 -m experiments.evaluate --alg_name=ROME --hparams_fname=default.json
```

Running the following command will yield `dict` run summaries:
```bash
python3 -m experiments/summarize --alg_name=ROME --run_name=run_001
``` -->

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
>>>>>>> BASE      (9df1dd Updates for releasing the repository)
