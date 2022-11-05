import argparse
import json
import os
import shutil
import collections
import time
from google.cloud import storage
from pathlib import Path
from typing import Tuple, Union
from contextlib import nullcontext

import torch
import pandas as pd
import numpy as np
from scipy.stats import hmean
from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

from baselines.efk import EFKHyperParams, EfkRewriteExecutor
from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.kn import KNHyperParams, apply_kn_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    get_tfidf_vectorizer,
)
from experiments.causal_trace import ModelAndTokenizer, predict_token
from experiments.causal_trace import layername, corrupted_forward_pass, find_token_range, make_inputs, simple_make_inputs
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.fewshot_utils import predict_model, fewshot_accuracy_sum, score_from_batch
from util.generate import generate_fast
from util.globals import *

ALG_DICT = {
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "KN": (KNHyperParams, apply_kn_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
    "KE": (EFKHyperParams, EfkRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}

CODE_DIR='/home/peterhase/belief-loc'
BASE_DIR='/home/peterhase'

def get_override_hparams(args, window_size, central_layer, alg_name):
  # embeddings are being FTed
  if central_layer == -1:
      assert alg_name == 'FT'
      return_dict = {
          'lr': 1e-3,
          'num_steps': 100,
          'norm_constraint': .01,
          'layers': [-1],
      }
      if window_size > 1:
          print("IGNORING WINDOW SIZE FOR TUNING EMBEDDINGS")
  # window size 1 approach
  elif window_size == 1:
    return_dict = {'layers' : [central_layer]}
    # weight norm constraints for each method
    if alg_name == "FT":
        if args.fact_erasure:
            return_dict['norm_constraint'] = 5e-5
        if args.fact_amplification:
            return_dict['norm_constraint'] = 5e-5
        elif args.fact_forcing:
            return_dict['norm_constraint'] = 1e-4
        elif args.tracing_reversal:
            return_dict['norm_constraint'] = 1e-3
        else:
            return_dict['norm_constraint'] = 1e-4
  # hack for applying ROME to multiple 3 layers
  elif window_size == 3 and alg_name == 'ROME':
    # budget the window size so that we're never editing fewer than three layers
    # hardcoding for now
    layers = [central_layer-1, central_layer, central_layer+1]
    if min(layers) < 0:
        offset = min(layers)
        layers = [layer - offset for layer in layers]
    if max(layers) > max(central_layers):
        offset = max(layers) - num_layers
        layers = [layer - offset for layer in layers]
    return_dict = {
        'layers' : layers,
        'v_num_grad_steps': 4,
        'v_lr': 0.1
        }
  elif window_size > 1:
    layer = central_layer
    window = window_size
    # same layers logic as used in causal tracing. there is clipping at the edges of the network
    layers = list(range(
        max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
        ))
    return_dict = {'layers' : layers}
    if alg_name == "FT":
      return_dict['norm_constraint'] = 1e-5
  # method specific parameters
  # increase number of steps if noising the subject
  if args.fact_forcing:
    if alg_name == "FT":
        return_dict['num_steps'] = 50
    if alg_name == "ROME":
        return_dict['v_num_grad_steps'] = 50
  if args.weight_based_tracing:
    if alg_name == "FT":
        return_dict['num_steps'] = 500
        return_dict['lr'] = 1e-5
  return return_dict

def sweep_experiment_name(args, model_name, alg_name, ds_name, sweep_params):
  exp_name = f'{model_name}_{alg_name}_outputs_{ds_name}_editing_sweep'  
  for k,v in sweep_params.items():
    _v = str(v).replace(", ", "-")
    if _v == "-1":
        _v = "embeds"
    if _v == "-2":
        _v = "all"
    exp_name += f"_{k[:5]}-{_v}"
  if args.tracing_reversal:
    obj = '_trace-reverse'
  elif args.fact_forcing:
    obj = '_fact-forcing'
  elif args.fact_erasure:
    obj = '_fact-erasure'
  elif args.fact_amplification:
    obj = '_fact-amplification'
  elif args.weight_based_tracing:
    obj = '_weight-tracing'
  else:
    obj = ''
  return f'{exp_name}{obj}_n{args.dataset_size_limit}'

def ROME_experiment_name(args, model_name, alg_name, ds_name, hparams_to_add):
  exp_name = f'{model_name}/{alg_name}_outputs_{ds_name}'
  if args.tracing_reversal:
    hparams_to_add['trace-reverse'] = 'T'
  if args.fact_forcing:
    hparams_to_add['fact-forcing'] = 'T'
  if args.fact_erasure:
    hparams_to_add['min'] = 'T'
  if args.fact_amplification:
    hparams_to_add['ampfy'] = 'T'
  if args.weight_based_tracing:
    hparams_to_add['weight-based'] = 'T'
  for k,v in hparams_to_add.items():
    _v = str(v).replace(", ", "-")
    if _v == "-1":
        _v = "embeds"
    exp_name += f"_{k[:5]}-{_v}"
  return exp_name

def ROME_experiment_name_from_override_params(args, model_name, alg_name, ds_name, override_hparams, hparams_class):
  _model_name = model_name.replace('/', '_')
  params_path = os.path.join(f'{CODE_DIR}/hparams/', alg_name, f"{_model_name}.json")
  if alg_name == 'FT':
    params_path = params_path.replace('.json', '_constr.json')
  hparams = hparams_class.from_json(params_path)
  if override_hparams is not None:
      hparams.__dict__.update(override_hparams)
  important_hparam_names = override_hparams.keys() if override_hparams is not None else ['layers']
  important_hparams = {k:v for k,v in hparams.__dict__.items() if any([k==name for name in important_hparam_names])}
  exp_name = ROME_experiment_name(args,
                                  model_name.split('/')[-1],
                                  alg_name,
                                  ds_name,
                                  important_hparams)
  return exp_name

def make_editing_results_df(exp_name, n=1000):
  run_dir = os.path.join(f'{BASE_DIR}/results/', exp_name)
  dataframes = []
  printed = 0
  for case_id in range(n):
    case_result_path = os.path.join(run_dir, f"case_{case_id}.json")
    if not os.path.exists(case_result_path):
      if printed < 10:
        print("skipping ", case_result_path, " does not exist")
        printed+=1
      continue
    with open(case_result_path, 'r') as f:
      record = json.load(f)
    rewrite_data = record['requested_rewrite']
    prompt = rewrite_data['prompt'].format(rewrite_data['subject'])
    target = rewrite_data['target_true']['str']
    try:
        record_dict = {
            'case_id': [record['case_id']],
            'prompt': [prompt],
            'target': [target],
            'subject' : [rewrite_data['subject']],
            'request' : [rewrite_data['target_new']['str']],
            'request_baseline': [rewrite_data['request_baseline']]
        }
    except:
        print("skipping ", case_result_path, " missing basic info")
        continue
    cur_sum = collections.defaultdict(lambda: [])
    data = record
    # record difference in pre and post probs for target_new
    for data_type in ['rewrite', 'paraphrase', 'neighborhood']:
        post_prob = np.exp(-data['post'][f'{data_type}_prompts_probs'][0]['target_new'])
        pre_prob = np.exp(-data['pre'][f'{data_type}_prompts_probs'][0]['target_new'])
        cur_sum[f'{data_type}_prob_diff'] = post_prob - pre_prob
        cur_sum[f'{data_type}_pre_prob'] = pre_prob
        cur_sum[f'{data_type}_post_prob'] = post_prob
        erased_prop = (pre_prob - post_prob) / pre_prob
        erased_prop = np.max([0, erased_prop])
        recovered_prop = 1 - (1 - post_prob) / (1 - pre_prob)
        recovered_prop = np.max([0, recovered_prop])
        abs_diff = np.abs(post_prob-pre_prob)
        cur_sum[f'{data_type}_recovered'] = recovered_prop
        cur_sum[f'{data_type}_erased'] = erased_prop
        max_abs_diff = np.abs(pre_prob - .5) + .5
        if data_type != 'neighborhood':
            cur_sum[f'{data_type}_score'] = erased_prop if args.fact_erasure else recovered_prop
        else:
            cur_sum[f'{data_type}_score'] = 1 - abs_diff / max_abs_diff
    cur_sum["target_score"] = hmean([
        cur_sum['rewrite_score'], cur_sum['paraphrase_score'], cur_sum['neighborhood_score']
    ])
    # compute essence scores 
    if 'essence_score' in data["post"]:
        cur_sum[f"post_essence_ppl"] = data["post"]['essence_score']
        cur_sum[f"pre_essence_ppl"] = data["pre"]['essence_score']
        cur_sum['essence_ppl_diff'] = cur_sum['post_essence_ppl'] - cur_sum['pre_essence_ppl'] # lower is better
    # compute original ROME metrics
    for prefix in ["pre", "post"]:
        # Probability metrics for which new should be lower (better) than true
        for key in ["rewrite_prompts_probs", "paraphrase_prompts_probs"]:
            if prefix not in data or key not in data[prefix]:
                continue
            sum_key_discrete = f"{prefix}_{key.split('_')[0]}_success"
            sum_key_cont = f"{prefix}_{key.split('_')[0]}_diff"
            cur_sum[sum_key_discrete].append(
                np.mean(
                    [
                        x["request_baseline"] > x["target_new"]
                        for x in data[prefix][key]
                    ]
                )
            )
            cur_sum[sum_key_cont].append(
                np.mean(
                    [
                        np.exp(-x["target_new"]) - np.exp(-x["request_baseline"])
                        for x in data[prefix][key]
                    ]
                )
            )
        # Probability metrics for which true should be lower (better) than new
        sum_key_discrete = f"{prefix}_neighborhood_success"
        sum_key_cont = f"{prefix}_neighborhood_diff"
        key = "neighborhood_prompts_probs"
        if prefix in data and key in data[prefix]:
            cur_sum[sum_key_discrete].append(
                np.mean(
                    [
                        x["request_baseline"] < x["target_new"]
                        for x in data[prefix][key]
                    ]
                )
            )
            cur_sum[sum_key_cont].append(
                np.mean(
                    [
                        np.exp(-x["request_baseline"]) - np.exp(-x["target_new"])
                        for x in data[prefix][key]
                    ]
                )
            )
        # zsRE evaluation metrics
        for key in ["rewrite", "paraphrase", "neighborhood"]:
            sum_key = f"{prefix}_{key}_acc"
            key = f"{key}_prompts_correct"
            if prefix not in data or key not in data[prefix]:
                continue
            cur_sum[sum_key].append(np.mean(data[prefix][key]))
        # get harmonic mean averages per point
        for prefix in ["pre", "post"]:
            for k_efficacy, k_generalization, k_specificity in [(
                    f"{prefix}_rewrite_success",
                    f"{prefix}_paraphrase_success",
                    f"{prefix}_neighborhood_success",
                ),
                (
                    f"{prefix}_rewrite_acc",
                    f"{prefix}_paraphrase_acc",
                    f"{prefix}_neighborhood_acc",
                )]:
                  if k_generalization in cur_sum and k_specificity in cur_sum:
                      cur_sum[f"{prefix}_score"] = hmean([
                                  cur_sum[k_efficacy][0],
                                  cur_sum[k_generalization][0],
                                  cur_sum[k_specificity][0]]
                      )
    # add ROME metrics to record_dict and append to dataframes
    record_dict.update(cur_sum)
    df = pd.DataFrame(record_dict)
    dataframes.append(df)
  if len(dataframes) > 0:
    return_df = pd.concat(dataframes)
  else:
    return_df = pd.DataFrame()
  return return_df


def get_subject_noising_function(model, e_range, hparams, embed_layername):
    # define noise embeddings function
    prng = np.random.RandomState(1) 
    # define function that noises embeddings at tokens_to_mix indices
    def noise_embeddings_f(x, layer):
        # skip noising if seq is a single token (must be bos/eos for open-ended generation)
        if (x.shape[1] == 1):
            return x
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if e_range is not None:
                b, e = e_range
                embeds_noise = torch.from_numpy(prng.randn(x.shape[0], e - b, x.shape[2])).to(x.device)
                x[:, b:e] += hparams.editing_noise * embeds_noise
            # print("added noise to embeds: ", embeds_noise)
            return x
        else:
            return x
    return noise_embeddings_f

def main(
    args,
    alg_name: str,
    model_name: Union[str, Tuple],
    ds_name: str,
    dataset_size_limit: int,
    do_essence_tests: bool,
    skip_generation_tests: bool,
    conserve_memory: bool,
    mt=None,
    verbose=False,
    override_hparams=None,
    overwrite=False,
    correctness_check=False,
    target_prob_check=0,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Get run hyperparameters
    _model_name = model_name.replace('/', '_')
    params_path = os.path.join(f'{CODE_DIR}/hparams/', alg_name, f"{_model_name}.json")
    if alg_name == 'FT':
      params_path = params_path.replace('.json', '_constr.json')
    hparams = params_class.from_json(params_path)
    args.hparams = hparams
    if override_hparams is not None:
      hparams.__dict__.update(override_hparams)
    print(f"Executing {alg_name} with parameters {hparams}")

    # Determine run directory
    important_hparam_names = override_hparams.keys() if override_hparams is not None else ['layers']
    important_hparams = {k:v for k,v in hparams.__dict__.items() if any([k==name for name in important_hparam_names])}
    exp_name = ROME_experiment_name(args,
                                    model_name.split('/')[-1],
                                    alg_name,
                                    ds_name,
                                    important_hparams)
    run_dir = f'{BASE_DIR}/results/{exp_name}'
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results will be stored at {run_dir}")
    # copy hparams to results dir
    copy_to_path =  os.path.join(run_dir, 'hparams.json')
    if not os.path.exists(copy_to_path):
        shutil.copyfile(params_path, copy_to_path)
    
    # Instantiate vanilla model
    if mt is None:
      print("Instantiating model")
      model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
      tok = AutoTokenizer.from_pretrained(model_name)
      tok.pad_token = tok.eos_token
    else:
      model, tok = mt.model, mt.tokenizer
      tok.pad_token = tok.eos_token

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR)
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, size=dataset_size_limit, tok=tok)
    # Iterate through dataset
    for record in ds:
        case_id = record["case_id"] if 'case_id' in record else 'known_id'
        case_result_path = os.path.join(run_dir, f"case_{case_id}.json")
        rewrite_this_point = overwrite or not os.path.exists(case_result_path)
        if case_id == 1531: # weird memory issue here
            continue
        if rewrite_this_point:
            print("Starting point: ", case_id)
            # print info for this point
            request = record["requested_rewrite"]
            subject = record["requested_rewrite"]['subject']
            prompt = request['prompt'].format(subject)
            request['full_prompt'] = prompt
            target_true = request['target_true']['str']
            paraphrase_prompts = record["paraphrase_prompts"]
            neighborhood_prompts = record["neighborhood_prompts"]
            if verbose:
                print("Updating point:"
                    f" orig update: [{prompt}] -> [{request['target_new']['str']}]"
                    f"\n True label: {target_true}"
                    f"\n Paraphrases: {paraphrase_prompts[:2]}"
                    f"\n Neighbors: {neighborhood_prompts[:2]}")

            # check if we should skip based on correctness and probability checks
            if correctness_check or target_prob_check > 0:
                is_correct, meets_target_prob = True, True
                if correctness_check:
                    gen_batch = simple_make_inputs(tok, prompts=[prompt])
                    samples, scores, _ = predict_model(mt, 
                                            [prompt], 
                                            answers=None, 
                                            trigger_phrase=None, 
                                            max_decode_steps=36)
                    is_correct = fewshot_accuracy_sum(samples, [target_true])
                if target_prob_check > 0:
                    preds, scores, _ = predict_model(mt, [prompt], answers=[target_true])
                    meets_target_prob = scores[0].item() > target_prob_check
                if not (is_correct and meets_target_prob):
                    if verbose:
                        print(" Skipping this point due to it being incorrect or not meeting the minimum target prob.")
                        if target_prob_check > 0: 
                            print(f" Target prob: {scores[0].item():.4f}")
                            print(f" Pred: {preds}")
                    continue

            # generate essence_texts for evaluation if needed
            if do_essence_tests or not skip_generation_tests:
                essence_prompt = "{} is a".format(subject)
                # second condition here checks that subject appears verbatim in first 200 characters of essence text, which is necessary later on
                # essence_texts = snips.names_to_samples[subject]
                # if subject == 'Inner Circle railway line':
                    # print("debug the condition below for Inner Circle railway line")
                # if len(essence_texts) == 0 or not all([subject in essence_text[:200] for essence_text in essence_texts]):
                if verbose:
                    print("GENERATING ESSENCE TEXTS")
                essence_texts = generate_fast(
                    model,
                    tok,
                    [essence_prompt],
                    n_gen_per_prompt=5,
                    max_out_len=100,
                )
                snips.names_to_samples[subject] = essence_texts
                # snips.names_to_samples[subject].extend(essence_texts)
                # elif verbose:
                    # print("using wikipedia essence texts")
                if verbose:
                    for text in snips.names_to_samples[request['subject']][:2]:
                        print(f" Essence text: {text[:200]}")

            # adjust targets and define 'request_baseline' based on objectives. note model does not necesarily predict 'request_baseline' value before rewriting
            num_noise_samples = 10 if args.fact_forcing else 1
            e_range = find_token_range(tok, substring=subject, prompt_str=prompt)
            request['e_range'] = e_range
            prior_prob = None
            # make noise embeddings_f
            embed_layername = layername(model, 0, 'embed')
            noise_embeddings_f = get_subject_noising_function(model, e_range, hparams, embed_layername)
            if args.tracing_reversal:
                gen_batch = simple_make_inputs(tok, prompts=[prompt] * (num_noise_samples))
                with torch.no_grad(), nethook.TraceDict(model, [embed_layername], edit_output=noise_embeddings_f) as td:
                    essence_texts = generate_fast(
                        model,
                        tok,
                        [prompt],
                        n_gen_per_prompt=1,
                        max_out_len=12,
                    )
                    new_target = essence_texts[0]
                # noised_pred_tokens = tok.encode(essence_texts[0])
                # _, noised_pred_id = corrupted_forward_pass(mt.model, None, gen_batch, tokens_to_mix=e_range, noise=hparams.editing_noise)
                # noised_pred_token = tok.decode([noised_pred_id])
                request['request_baseline'] = request['target_true']['str']
                request['target_new']['str'] = new_target
                request['target_new']['id'] = 'noised-input'
                if verbose:
                    score_batch = make_inputs(tok, [prompt], targets=[new_target])
                    init_target_prob = score_from_batch(model, score_batch)
                    print(f" NEW TARGET PREDICTION: {new_target}")
                    print(f" with init pred prob: {init_target_prob.item():.4f}")
            elif args.fact_erasure:
                batch = make_inputs(mt.tokenizer, prompts=[prompt] * num_noise_samples, targets=[target_true] * num_noise_samples)
                prior_prob = corrupted_forward_pass(mt.model, batch, None, tokens_to_mix=e_range, noise=hparams.editing_noise)
                prior_prob = prior_prob.item()
                request['request_baseline'] = mt.tokenizer.eos_token # arbitrary token, won't use these metrics anyway
                request['target_new'] = request['target_true']
            elif args.fact_amplification:
                request['request_baseline'] = mt.tokenizer.eos_token # arbitrary token, won't use these metrics anyway
                request['target_new'] = request['target_true']
            elif args.fact_forcing or args.weight_based_tracing:
                gen_batch = simple_make_inputs(tok, prompts=[prompt] * (num_noise_samples))
                _, noised_pred_id = corrupted_forward_pass(mt.model, None, gen_batch, tokens_to_mix=e_range, noise=hparams.editing_noise)
                noised_pred_token = tok.decode([noised_pred_id])
                request['request_baseline'] = noised_pred_token
                request['target_new'] = request['target_true']
            else:
                request['request_baseline'] = request['target_true']['str']
            if verbose:
                print(" request baseline: ", request['request_baseline'])
                
            # get additional functions and variables based on objectives
            # get hidden representations from corrupted+uncorrupted forward passes to use as targets for weight editing
            if args.weight_based_tracing:
                last_subj_idx = e_range[1]
                with torch.enable_grad():
                    # corrupted forward pass. corrupted_hidden_states will be of shape [n_layers, num_noise_samples, seq_len, hidden_dim]
                    gen_batch = simple_make_inputs(tok, prompts=[prompt] * num_noise_samples)
                    gen_batch['output_hidden_states'] = True
                    _, _, corrupted_hidden_states = corrupted_forward_pass(model, None, gen_batch, tokens_to_mix=e_range, noise=hparams.editing_noise, output_hidden_states=True)
                    corrupted_hidden_states = torch.stack([corrupted_hidden_states[layer+1] for layer in hparams.layers], dim=0)
                    # clean forward pass
                    gen_batch = simple_make_inputs(tok, prompts=[prompt])
                    clean_hidden_states = model(**gen_batch, output_hidden_states=True).hidden_states
                    clean_hidden_states = torch.stack([clean_hidden_states[layer+1] for layer in hparams.layers], dim=0)
                # splice uncorrupted hidden_states into corrupted_hidden_states where they are restored. automatically broadcast across num_noise_samples dimension
                hidden_state_supervision = corrupted_hidden_states
                hidden_state_supervision[:,:,last_subj_idx,:] = clean_hidden_states[:,:,last_subj_idx,:]
            else:
                hidden_state_supervision = None

            # Compute weight changes + record weights that changed
            start = time.time()
            args_conserve_memory = (
                dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
                if conserve_memory
                else dict()
            )
            with torch.enable_grad(), nethook.TraceDict(model, [embed_layername], edit_output=noise_embeddings_f) if args.fact_forcing else nullcontext() as td:
              edited_model, weights_copy = apply_algo(
                  args,
                  model,
                  tok,
                  [request],
                  hparams,
                  copy=False,
                  return_orig_weights=True,
                  num_noise_samples=num_noise_samples,
                  prior_prob=prior_prob,
                  hidden_state_supervision=hidden_state_supervision,
                  **args_conserve_memory,
              )
            exec_time = time.time() - start
            print("Execution took", exec_time)

            # Execute evaluation suite
            start = time.time()
            with torch.no_grad(): 
                metrics = {
                    "case_id": case_id,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": ds_eval_method(args, edited_model, tok, record, snips, vec, skip_generation_tests),
                }
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to("cuda")
                metrics["pre"] = ds_eval_method(args, model, tok, record, snips, vec, skip_generation_tests)
                metrics['prior_prob'] = prior_prob

            print("Evaluation took", time.time() - start)
            # Dump metrics in .json
            with open(case_result_path, "w") as f:
                json.dump(metrics, f, indent=1)
            print('\n')
        else:
            if verbose:
              print(f"skipping {case_result_path}, already run")
            else:
              pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["ROME", "FT", "KN", "MEND", "KE"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--edit_layer",
        type=int,
        default=0,
        help="Layer at which to edit the model weights. Set to -2 to defer to hparam sweep params below",
    )
    parser.add_argument(
        "--ds_name",
        choices=["cf", "zsre"],
        default="cf",
        help="Dataset to perform evaluations on. Either CounterFact (cf) or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--window_sizes",
        type=str,
        default='1',
        help="Window sizes separted by spaces to use for editing method",
    )
    parser.add_argument(
        "--dataset_size_limit",
        "-n",
        type=int,
        default=1000,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--norm_constraint",
        type=float,
        default=1e-4,
        help="l0 norm constraint on constrained finetuning",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previous experiment results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="More printing during editing",
    )
    parser.add_argument(
        "--do_essence_tests",
        type=int,
        default=1,
        choices=[0,1],
        help="Do the essence drift generation test regardless of args.skip_generation_tests",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--tracing_reversal",
        action="store_true",
        help="Rather than changing output from target_true to target_new, change it to the prediction obtained from the noised causal tracing input",
    )
    parser.add_argument(
        "--fact_forcing",
        action="store_true",
        help="Rather than change o-true to o-new for (s,r,.) input, change o-noise to o-true for (s-noise, r,.) input",
    )
    parser.add_argument(
        "--fact_erasure",
        action="store_true",
        help="See paper for description",
    )
    parser.add_argument(
        "--fact_amplification",
        action="store_true",
        help="See paper for description",
    )
    parser.add_argument(
        "--weight_based_tracing",
        action="store_true",
        help="See paper for description",
    )
    parser.add_argument(
        "--correctness_filter",
        type=int,
        default=0,
        choices=[0,1],
        help="Only eval on points with correct generations or p(target_true) >= .1",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=1,
        choices=[0,1],
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.set_defaults(skip_generation_tests=True, conserve_memory=True)
    args = parser.parse_args()

    # set device
    device = torch.device(f"cuda:{args.gpu}")
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # experiment checks
    if args.fact_erasure:
        assert args.correctness_filter, "only erase known facts"

    # load model
    if args.run:
        torch.set_grad_enabled(False)
        model_name = args.model_name
        torch_dtype = torch.float16 if '20b' in model_name else None
        mem_usage = True
        print("Loading model...")
        if '20b' not in model_name:
            mt = ModelAndTokenizer(model_name, low_cpu_mem_usage=mem_usage, torch_dtype=torch_dtype)
            torch.cuda.empty_cache()
            mt.model.eval().cuda()
            mt.tokenizer.add_special_tokens({'pad_token' : mt.tokenizer.eos_token})
            # mt.tokenizer.pad_token = mt.tokenizer.eos_token
            # mt.tokenizer.pad_token_id = mt.tokenizer.eos_token_id
        else:
            model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", 
                                                        device_map={
                                                            'embed_out' : 0,
                                                            'gpt_neox.embed_in' : 0,
                                                            'gpt_neox.layers': 1,
                                                            'gpt_neox.final_layer_norm' : 0,
                                                        },
                                                        low_cpu_mem_usage=mem_usage,
                                                        torch_dtype=torch_dtype)
            torch.cuda.empty_cache()
            model.eval().cuda()
            tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
            mt = ModelAndTokenizer(model=model, tokenizer=tokenizer, torch_dtype=torch_dtype)
            mt.tokenizer.add_special_tokens({'pad_token' : mt.tokenizer.eos_token})
        # num_layers = mt.num_layers

    # set experiment args
    RUN_EXPERIMENT = args.run # set to false to just collect results
    num_points = args.dataset_size_limit
    alg_name = args.alg_name
    model_name = args.model_name
    assert alg_name in ["FT", "ROME"]
    hparams_class = FTHyperParams if alg_name == "FT" else ROMEHyperParams
    ds_name = args.ds_name
    window_sizes = [int(x) for x in args.window_sizes.split()]
    if 'gpt2' in model_name:
        central_layers = list(range(0, 48, 4)) + [17, 47]
        num_layers = 48
    if '6B' in model_name:
        central_layers = list(range(0, 28, 4)) + [5, 27]
        num_layers = 28
    if alg_name == 'FT' and 1 in window_sizes and not args.fact_forcing:
        central_layers = [-1] + central_layers
    if args.edit_layer > -2:
        central_layers = [args.edit_layer]
    if alg_name == "ROME":
        central_layers = np.setdiff1d(central_layers, [24, 27])
    print("Starting sweep with hparams:")
    print("- window_sizes: ", window_sizes)
    print("- central_layers: ", central_layers)

    # main experiment loop
    results_dfs = []
    for window_size in window_sizes:
        for central_layer in central_layers:
            override_hparams = get_override_hparams(args, window_size, central_layer, alg_name)
            if RUN_EXPERIMENT:
                main(
                    args,
                    alg_name=alg_name,
                    model_name=model_name,
                    ds_name=ds_name,
                    dataset_size_limit=num_points,
                    do_essence_tests=args.do_essence_tests,
                    skip_generation_tests=args.skip_generation_tests,
                    conserve_memory=args.conserve_memory,
                    mt=mt,
                    override_hparams=override_hparams,
                    verbose=args.verbose,
                    overwrite=args.overwrite,
                    correctness_check=args.correctness_filter,
                    target_prob_check=.02 if args.correctness_filter and args.fact_erasure else 0
                )
            # accumulate results
            exp_name = ROME_experiment_name_from_override_params(args, model_name, alg_name, ds_name, override_hparams, hparams_class)
            editing_results_df = make_editing_results_df(exp_name, n=num_points)
            editing_results_df['edit_method'] = alg_name
            editing_results_df['edit_central_layer'] = central_layer
            editing_results_df['edit_window_size'] = window_size
            results_dfs.append(editing_results_df)
    
    # combine and save results
    results_df = pd.concat(results_dfs)
    _model_name = model_name.split('/')[-1]
    sweep_params = {'ws': window_sizes, 'layers': args.edit_layer}
    ovr_exp_name = sweep_experiment_name(args, _model_name, alg_name, ds_name, sweep_params)
    file_name = f'{ovr_exp_name}.csv'
    save_path = f'{BASE_DIR}/results/{file_name}'
    results_df.to_csv(save_path, index=False)
    # upload results csv to google bucket
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('research-brain-belief-localization-xgcp')
    blob = bucket.blob(f'output/{file_name}')
    blob.upload_from_filename(save_path)

    print(f"saving csv at {save_path}...")
    print("results shape: ", results_df.shape)
    # metrics = ['rewrite_prob_diff', 'rewrite_post_prob', 'rewrite_score', 'paraphrase_prob_diff', 'paraphrase_post_prob', 'paraphrase_score', 'neighborhood_prob_diff', 'neighborhood_score']
    metrics = ['post_rewrite_success', 'rewrite_prob_diff', 'rewrite_post_prob', 'rewrite_score', 'post_paraphrase_success', 'paraphrase_prob_diff', 'paraphrase_post_prob', 'paraphrase_score', 'post_neighborhood_success', 'neighborhood_prob_diff', 'neighborhood_score']
    # metrics = ['rewrite_score', 'paraphrase_score', 'neighborhood_score', 'target_score']
    # if args.fact_erasure or args.fact_amplification or args.fact_forcing or args.weight_based_tracing:
    #     metrics = ['rewrite_prob_diff', 'paraphrase_prob_diff', 'neighborhood_prob_diff', 'essence_ppl_diff', 'post_score', 'erasure_loss']
    # else:
    #     metrics = ['post_rewrite_success', 'post_rewrite_diff', 'post_neighborhood_success', 'post_neighborhood_diff', 'post_paraphrase_success', 'post_paraphrase_diff', 'essence_ppl_diff', 'post_score']
    if len(window_sizes) == 1 and len(central_layers) == 1:
        print("\nfinal metrics: ")
        for metric in metrics:
            if metric in results_df.columns:
                avg_val = np.mean(results_df.loc[:,metric])
                print(f" {metric:.20s}: {avg_val:.3f}")
            else:
                print(f" missing {metric}")

