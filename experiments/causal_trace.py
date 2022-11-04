import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
import unidecode
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import KnownsDataset
from rome.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)
from util.fewshot_utils import score_from_batch
from util import nethook
from util.globals import DATA_DIR
from util.runningstats import Covariance, tally

def main():
    parser = argparse.ArgumentParser(description="Causal Tracing")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    def parse_noise_rule(code):
        if code in ["m", "s"]:
            return code
        elif re.match("^[uts][\d\.]+", code):
            return code
        else:
            return float(code)

    aa(
        "--model_name",
        default="gpt2-xl",
        choices=[
            "gpt2-xl",
            "EleutherAI/gpt-j-6B",
            "EleutherAI/gpt-neox-20b",
            "gpt2-large",
            "gpt2-medium",
            "gpt2",
        ],
    )
    aa("--fact_file", default=None)
    aa("--output_dir", default="results/{model_name}/causal_trace")
    aa("--noise_level", default="s3", type=parse_noise_rule)
    aa("--replace", default=0, type=int)
    args = parser.parse_args()

    modeldir = f'r{args.replace}_{args.model_name.replace("/", "_")}'
    modeldir = f"n{args.noise_level}_" + modeldir
    output_dir = args.output_dir.format(model_name=modeldir)
    result_dir = f"{output_dir}/cases"
    pdf_dir = f"{output_dir}/pdfs"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Half precision to let the 20b model fit.
    torch_dtype = torch.float16 if "20b" in args.model_name else None

    mt = ModelAndTokenizer(args.model_name, torch_dtype=torch_dtype)

    if args.fact_file is None:
        knowns = KnownsDataset(DATA_DIR)
    else:
        with open(args.fact_file) as f:
            knowns = json.load(f)

    noise_level = args.noise_level
    uniform_noise = False
    if isinstance(noise_level, str):
        if noise_level.startswith("s"):
            # Automatic spherical gaussian
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            sd = collect_embedding_std(
                mt, [k["subject"] for k in knowns]
            )
            noise_level = factor * sd
            print(f"Using noise_level {noise_level} to match model stdev {sd} times {factor}")
        elif noise_level == "m":
            # Automatic multivariate gaussian
            noise_level = collect_embedding_gaussian(mt)
            print(f"Using multivariate gaussian to match model noise")
        elif noise_level.startswith("t"):
            # Automatic d-distribution with d degrees of freedom
            degrees = float(noise_level[1:])
            noise_level = collect_embedding_tdist(mt, degrees)
        elif noise_level.startswith("u"):
            uniform_noise = True
            noise_level = float(noise_level[1:])

    for knowledge in tqdm(knowns):
        known_id = knowledge["known_id"]
        for kind in None, "mlp", "attn":
            kind_suffix = f"_{kind}" if kind else ""
            filename = f"{result_dir}/knowledge_{known_id}{kind_suffix}.npz"
            if not os.path.isfile(filename):
                result = calculate_hidden_flow(
                    mt,
                    knowledge["prompt"],
                    knowledge["subject"],
                    expect=knowledge["attribute"],
                    kind=kind,
                    noise=noise_level,
                    uniform_noise=uniform_noise,
                    replace=args.replace,
                )
                numpy_result = {
                    k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                    for k, v in result.items()
                }
                np.savez(filename, **numpy_result)
            else:
                numpy_result = np.load(filename, allow_pickle=True)
            if not numpy_result["correct_prediction"]:
                tqdm.write(f"Skipping {knowledge['prompt']}")
                continue
            plot_result = dict(numpy_result)
            plot_result["kind"] = kind
            pdfname = f'{pdf_dir}/{str(numpy_result["answer"]).strip()}_{known_id}{kind_suffix}.pdf'
            if known_id > 200:
                continue
            plot_trace_heatmap(plot_result, savepdf=pdfname)

def corrupted_forward_pass(
    model,            # The model
    batch,            # A set of inputs. Assumed to be the same input text repeated num_noise_samples times
    gen_batch,        # Set of inputs tokenized into pytorch batch without targets, used to get predicted output from noised subject input. Assumed to be the same input text repeated
    tokens_to_mix,    # Range of tokens to corrupt (begin, end)
    noise=0.1,        # Level of noise to add
    output_hidden_states=False,
    ):
    prng = np.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    embed_layername = layername(model, 0, 'embed')
    assert batch is None or gen_batch is None
    # define function that noises embeddings at tokens_to_mix indices
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                embeds_noise = torch.from_numpy(prng.randn(x.shape[0], e - b, x.shape[2])).to(x.device)
                x[:, b:e] += noise * embeds_noise
            # print("added noise to embeds: ", embeds_noise)
            return x
        else:
            return x
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername],
        edit_output=patch_rep
    ):
        if batch is not None:
            probs = score_from_batch(model, batch)
            outputs = probs.mean() # average over noise samples
            return outputs
        elif gen_batch is not None:
            if output_hidden_states:
                gen_batch['output_hidden_states'] = True
            pure_noise_outputs = model(**gen_batch)
            logits = pure_noise_outputs.logits
            probs = torch.softmax(logits[:, -1], dim=1).mean(dim=0) # average over noise samples
            noised_pred_id = torch.argmax(probs)
            pred_prob = probs[noised_pred_id]
            outputs = (pred_prob, noised_pred_id,)
            if output_hidden_states:
                outputs += (pure_noise_outputs.hidden_states,)
            return outputs


def trace_with_repatch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
):
    rs = np.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**inp)
            if first_pass:
                first_pass_trace = td

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs

def trace_with_patch(
    model,            # The model
    batch,            # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    pred_id,          # token id of answer probabilities to collect
    tokens_to_mix,    # Range of tokens to corrupt (begin, end)
    noise=0.1,        # Level of noise to add
    trace_layers=None # List of traced outputs to return
):
    prng = np.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, 'embed')
    
    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] +
            list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep
    ) as td:
        if 'target_indicators' not in batch:
          outputs_exp = model(**batch)
          assert pred_id is not None, "no targets provided, need to specify pred_id"
          # get the predicted probability for the pred_id token of interest, averaged across noise samples
          avg_corrupted_probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[pred_id]
          outputs = avg_corrupted_probs
        else:
          probs = score_from_batch(model, batch)
          outputs = probs[1:].mean() # probs[0] is uncorrupted pred probs

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2)
        return outputs, all_traced

    # return outputs.item()
    return outputs


def trace_important_states(model, num_layers, batch, e_range, pred_id=None, noise=0.1):
    if 'target_indicators' in batch:
      ntoks = batch["input_ids"].shape[1] - batch["target_indicators"].sum(-1)[0]
    else:
      ntoks = batch["input_ids"].shape[1]
    table = []
    start_token_idx = e_range[0]
    for tnum in range(start_token_idx, ntoks):
        row = []
        for layer in range(0, num_layers):
            print(f"tracing token {tnum}, layer {layer}", end='\r')
            r = trace_with_patch(
                model,
                batch,
                [(tnum, layername(model, layer))],
                pred_id,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model, num_layers, batch, e_range, pred_id=None, kind=None, window=10, noise=0.1, 
):
    if 'target_indicators' in batch:
      ntoks = batch["input_ids"].shape[1] - batch["target_indicators"].sum(-1)[0]
    else:
      ntoks = batch["input_ids"].shape[1]
    table = []
    start_token_idx = e_range[0]
    for tnum in range(start_token_idx, ntoks):
        row = []
        for layer in range(0, num_layers):
            print(f"tracing token {tnum}, layer {layer} for module type {kind}", end='\r')
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model, batch, layerlist, pred_id, tokens_to_mix=e_range, noise=noise, 
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def calculate_hidden_flow(
    mt, prompt, subject, target, samples=10, noise=0.1, window=10, output_type='probs', kind=None,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.

    Args
      target: str output to be explained
    """
    special_token_ids = [mt.tokenizer.eos_token_id, mt.tokenizer.bos_token_id, mt.tokenizer.pad_token_id]
    assert isinstance(prompt, str)
    if target is None:
        preds, scores, _ = predict_model(mt, [prompt], max_decode_steps=2, score_if_generating=True)
    else:
        preds, scores, query_inputs = predict_model(mt, [prompt], answers=[target])
    answer = preds[0]
    base_score = scores[0].item()
    pred_id = None 
    batch_size = (samples+1)
    batch = make_inputs(mt.tokenizer, prompts=[prompt] * batch_size, targets=[answer] * batch_size)
    e_range = find_token_range(mt.tokenizer, substring=subject, prompt_str=prompt)
    low_score = trace_with_patch(mt.model, batch, [], pred_id, tokens_to_mix=e_range, noise=noise)
    if not kind:
        differences = trace_important_states(
            mt.model, mt.num_layers, batch, e_range, pred_id, noise=noise,
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            batch,
            e_range,
            pred_id,
            noise=noise,
            window=window,
            kind=kind,
        )
    differences = differences.detach().cpu().squeeze()
    individual_tokens = [mt.tokenizer.decode([tok]) for idx, tok in enumerate(batch['query_ids'][0]) if (not tok in special_token_ids)]
    # make a few last things to add to dict
    if '\n' in individual_tokens: # assume prompting with examples separated by '\n'
      reversed_labels = list(reversed(individual_tokens))
      last_sep_idx = len(individual_tokens) - reversed_labels.index('\n') - 1
      test_tokens = individual_tokens[(last_sep_idx+1):]
    else:
      test_tokens = individual_tokens
    return dict(
        scores=differences,
        low_score=low_score.item(),
        base_score=base_score,
        high_score=max(base_score, differences.max().item()),
        input_tokens=individual_tokens,
        test_input_tokens=test_tokens,
        test_input_str=prompt.split('\n')[-1],
        subject_range=e_range,
        answer=answer,
        window=window,
        kind=kind or "",
    )


def get_high_and_low_scores(
    mt, prompt, subject, target, samples=10, noise=0.1,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.

    Args
      target: str output to be explained
    """
    special_token_ids = [mt.tokenizer.eos_token_id, mt.tokenizer.bos_token_id, mt.tokenizer.pad_token_id]
    assert isinstance(prompt, str)
    if target is None:
        preds, scores, _ = predict_model(mt, [prompt], max_decode_steps=2, score_if_generating=True)
    else:
        preds, scores, query_inputs = predict_model(mt, [prompt], answers=[target])
    answer = preds[0]
    base_score = scores[0].item()
    pred_id = None 
    batch_size = (samples+1)
    batch = make_inputs(mt.tokenizer, prompts=[prompt] * batch_size, targets=[answer] * batch_size)
    e_range = find_token_range(mt.tokenizer, substring=subject, prompt_str=prompt)
    low_score = trace_with_patch(mt.model, batch, [], pred_id, tokens_to_mix=e_range, noise=noise)
    high_score = scores[0]
    return high_score.item(), low_score.item()


def predict_model(mt, 
                query_inputs,
                answers=None, 
                trigger_phrase=None,
                max_decode_steps=None,
                score_if_generating=False):
  assert not isinstance(query_inputs, str), "provide queries as list"
  with torch.no_grad():
    generate_and_score = (answers is None)
    batch = make_inputs(mt.tokenizer, query_inputs, targets=answers)
    if generate_and_score:
      pad_token_id = mt.tokenizer.pad_token_id
      pad_token_id = pad_token_id if pad_token_id else 0
      outputs = mt.model.generate(**batch, do_sample=False, max_new_tokens=max_decode_steps,
                                  pad_token_id=pad_token_id)
      outputs = [list(filter(lambda x: x != pad_token_id, output)) for output in outputs]
      preds = [mt.tokenizer.decode(output) for output in outputs]
      preds = [pred.replace(query_input, "").strip() for pred, query_input in zip(preds, query_inputs)]     
      # for some reason huggingface generate not giving generation probs, so we recalculate
      if score_if_generating: 
        batch = make_inputs(mt.tokenizer, query_inputs, targets=preds)
        scores = score_from_batch(mt.model, batch)
      else:
        scores = -100 * np.ones(len(preds))
    else:
      num_answers = len(answers)
      repeated_inputs = []
      repeated_answers = []
      for input in query_inputs:
        for answer in answers:
          repeated_inputs.append(input)
          repeated_answers.append(answer)
      batch = make_inputs(mt.tokenizer, repeated_inputs, repeated_answers)
      scores = score_from_batch(mt.model, batch)
      scores = scores.reshape(-1, num_answers)
      pred_ids = [torch.argmax(ex_answer_probs).item() for ex_answer_probs in scores]
      preds = [answers[pred_id] for pred_id in pred_ids]
  return preds, scores, query_inputs


class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
    ):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            assert model_name is not None
            model = AutoModelForCausalLM.from_pretrained(
                model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
            )
            nethook.set_requires_grad(False, model)
            model.eval().cuda()
        self.tokenizer = tokenizer
        self.model = model
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def layername(model, num, kind=None):
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "unknown transformer structure"


def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()


def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.1,
    uniform_noise=False,
    window=10,
    kind=None,
    savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt,
        prompt,
        subject,
        samples=samples,
        noise=noise,
        uniform_noise=uniform_noise,
        window=window,
        kind=kind,
    )
    plot_trace_heatmap(result, savepdf)


def plot_trace_heatmap(result, savepdf=None, show_plot=True, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    low_score = result["low_score"]
    base_score = result["base_score"]
    high_score = result["high_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    # offset by prompt length if there is a prompt with separator "\n"
    sep = '\n'
    subj_range = result['subject_range']
    if sep in labels:
      reversed_labels = list(reversed(labels))
      last_sep_idx = len(labels) - reversed_labels.index('\n') - 1
      new_labels = labels[(last_sep_idx+1):]
      offset_by = len(labels)-len(new_labels)
      labels = new_labels
      subj_range = (subj_range[0] - offset_by, subj_range[1] - offset_by)
    for i in range(*subj_range):
        labels[i] = labels[i] + "*"
    # if labels do not match differences, it's because we started restoring
    # at the subject token beginning, so pad the differences with low_score
    if len(labels) != differences.shape[0]:
      short_by = len(labels)  - differences.shape[0]
      low_score_padding = low_score*np.ones((short_by, differences.shape[1]))
      differences = np.concatenate((low_score_padding, differences), axis=0)
    assert len(labels) == differences.shape[0], "num tokens doesnt match differences size"
    v_size = 3.5 if len(labels) < 10 else 4.1
    h_size = 2.8 if differences.shape[1] < 30 else 3.2
    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(h_size, v_size), dpi=300)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],
            vmin=low_score,
            vmax=high_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not kind:
            title = "Impact of restoring state after corrupted input"
            xlab = f"single restored layer within {modelname}"
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            title = f"Impact of restoring {kindname} after corrupted input"
            xlab = f"center of interval of {window} restored {kindname} layers"
        xlab += f"\n orig prob: {round(base_score, 3)}, noise prob: {round(low_score, 3)}"
        ax.set_xlabel(xlab)
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title, x=.5, y=1.01)
            # ax.set_title(title, x=.5, y=1.1)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-.08, fontsize=10)
            # cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            if show_plot:
              plt.show() 
            plt.close()
        elif show_plot:
            plt.show()


def plot_all_flow(mt, prompt, subject=None):
    for kind in ["mlp", "attn", None]:
        plot_hidden_flow(mt, prompt, subject, kind=kind)


# Utilities for dealing with tokens
def make_inputs(tokenizer, prompts, targets=None, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    elif tokenizer.pad_token_id is not None:
        pad_id = tokenizer.pad_token_id
    else:
        pad_id = 0
    if targets is None:
      maxlen = max(len(t) for t in token_lists)
      input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
      attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
      return dict(
          input_ids=torch.tensor(input_ids).to(device),
          attention_mask=torch.tensor(attention_mask).to(device),
      )
    if targets is not None:
      target_lists = [tokenizer.encode(" " + t) for t in targets]
      maxlen = max(len(p) + len(t) for p, t in zip(token_lists, target_lists))
      combine_lists = [p + t for p, t in zip(token_lists, target_lists)]
      query_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
      input_ids = [[pad_id] * (maxlen - len(t)) + t for t in combine_lists]
      target_ids = [[pad_id] * (maxlen - len(t)) + t for t in target_lists]
      target_indicators = [[0] * (maxlen - len(t)) + [1] * len(t) for t in target_lists]
      attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in combine_lists]
      return dict(
          input_ids=torch.tensor(input_ids).to(device),
          query_ids=torch.tensor(query_ids).to(device),
          target_ids=torch.tensor(target_ids).to(device),
          target_indicators=torch.tensor(target_indicators).to(device),
          attention_mask=torch.tensor(attention_mask).to(device),
      )


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array=None, substring=None, prompt_str=None):
    if prompt_str is None:
      toks = decode_tokens(tokenizer, token_array)
    else:
      token_array = tokenizer.encode(prompt_str)
      toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    try:
      char_loc = whole_string.index(substring)
    except:
      assert prompt_str is not None
      # clean non-unicode characters
      prompt_str = unidecode.unidecode(prompt_str)
      token_array = tokenizer.encode(prompt_str)
      toks = decode_tokens(tokenizer, token_array)
      whole_string = "".join(toks)
      substring = unidecode.unidecode(substring)
      # clean punctuation spacing
      for punc in ['.', ',', '!']:
        substring = substring.replace(f" {punc}", punc)
        whole_string = whole_string.replace(f" {punc}", punc)
      # just return None if substring not in whole_string now
      if substring not in whole_string:
          return None
      char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)

def simple_make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )

def predict_token(mt, prompts, return_p=False):
    inp = simple_make_inputs(mt.tokenizer, prompts)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result

def predict_from_input(model, inp):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p

def collect_embedding_std(mt, subjects):
    alldata = []
    for s in subjects:
        inp = make_inputs(mt.tokenizer, [s])
        with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level


def get_embedding_cov(mt):
    model = mt.model
    tokenizer = mt.tokenizer

    def get_ds():
        ds_name = "wikitext"
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
        )
        try:
            maxlen = model.config.n_positions
        except:
            maxlen = 100  # Hack due to missing setting in GPT2-NeoX.
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    ds = get_ds()
    sample_size = 1000
    batch_size = 5
    filename = None
    batch_tokens = 100

    progress = lambda x, **k: x

    stat = Covariance()
    loader = tally(
        stat,
        ds,
        cache=filename,
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=0,
    )
    with torch.no_grad():
        for batch_group in loader:
            for batch in batch_group:
                batch = dict_to_(batch, "cuda")
                del batch["position_ids"]
                with nethook.Trace(model, layername(mt.model, 0, "embed")) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                stat.add(feats.cpu().double())
    return stat.mean(), stat.covariance()


def make_generator_transform(mean=None, cov=None):
    d = len(mean) if mean is not None else len(cov)
    device = mean.device if mean is not None else cov.device
    layer = torch.nn.Linear(d, d, dtype=torch.double)
    nethook.set_requires_grad(False, layer)
    layer.to(device)
    layer.bias[...] = 0 if mean is None else mean
    if cov is None:
        layer.weight[...] = torch.eye(d).to(device)
    else:
        _, s, v = cov.svd()
        w = s.sqrt()[None, :] * v
        layer.weight[...] = w
    return layer


def collect_embedding_gaussian(mt):
    m, c = get_embedding_cov(mt)
    return make_generator_transform(m, c)


def collect_embedding_tdist(mt, degree=3):
    # We will sample sqrt(degree / u) * sample, where u is from the chi2[degree] dist.
    # And this will give us variance is (degree / degree - 2) * cov.
    # Therefore if we want to match the sample variance, we should
    # reduce cov by a factor of (degree - 2) / degree.
    # In other words we should be sampling sqrt(degree - 2 / u) * sample.
    u_sample = torch.from_numpy(
        np.random.RandomState(2).chisquare(df=degree, size=1000)
    )
    fixed_sample = ((degree - 2) / u_sample).sqrt()
    mvg = collect_embedding_gaussian(mt)

    def normal_to_student(x):
        gauss = mvg(x)
        size = gauss.shape[:-1].numel()
        factor = fixed_sample[:size].reshape(gauss.shape[:-1] + (1,))
        student = factor * gauss
        return student

    return normal_to_student


if __name__ == "__main__":
    main()
