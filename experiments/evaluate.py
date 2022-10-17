import json
import os
import shutil
import collections
import time
from google.cloud import storage
from pathlib import Path
from typing import Tuple, Union

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
from experiments.causal_trace import ModelAndTokenizer
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
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

def get_override_hparams(window_size, central_layer, alg_name):
  if central_layer == -1:
      assert window_size == 1
      assert alg_name == 'FT'
      return_dict = {
          'lr': 5e-4,
          'num_steps': 100,
          'norm_constraint': .01,
          'layers': [-1],
      }
  elif window_size == 1:
    return_dict = {'layers' : [central_layer]}
    if alg_name == "FT":
      return_dict['norm_constraint'] = 1e-4
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
  else:
    layer = central_layer
    window = window_size
    # same layers logic as used in causal tracing. there is clipping at the edges of the network
    layers = list(range(
        max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
        ))
    return_dict = {'layers' : layers}
    if alg_name == "FT":
      return_dict['norm_constraint'] = 2e-4
  return return_dict

def ROME_experiment_name(model_name, alg_name, ds_name, hparams_to_add):
  exp_name = f'{model_name}/{alg_name}_outputs_{ds_name}'
  for k,v in hparams_to_add.items():
    _v = str(v).replace(", ", "-")
    if _v == "-1":
        _v = "embeds"
    exp_name += f"_{k[:5]}-{_v}"
  return exp_name

def ROME_experiment_name_from_override_params(model_name, alg_name, ds_name, override_hparams, hparams_class):
  _model_name = model_name.replace('/', '_')
  params_path = os.path.join(f'{CODE_DIR}/hparams/', alg_name, f"{_model_name}.json")
  if alg_name == 'FT':
    params_path = params_path.replace('.json', '_constr.json')
  hparams = hparams_class.from_json(params_path)
  if override_hparams is not None:
      hparams.__dict__.update(override_hparams)
  important_hparam_names = override_hparams.keys() if override_hparams is not None else ['layers']
  important_hparams = {k:v for k,v in hparams.__dict__.items() if any([k==name for name in important_hparam_names])}
  exp_name = ROME_experiment_name(model_name.split('/')[-1],
                                  alg_name,
                                  ds_name,
                                  important_hparams)
  return exp_name

def make_editing_results_df(exp_name, n=1000):
  run_dir = os.path.join(f'{BASE_DIR}/results/', exp_name)
  dataframes = []
  for case_id in range(n):
    case_result_path = os.path.join(run_dir, f"case_{case_id}.json")
    if not os.path.exists(case_result_path):
      print("skipping ", case_result_path, " does not exist")
      continue
    with open(case_result_path, 'r') as f:
      record = json.load(f)
    rewrite_data = record['requested_rewrite']
    prompt = rewrite_data['prompt'].format(rewrite_data['subject'])
    target = rewrite_data['target_true']['str']
    record_dict = {
        'case_id': [record['case_id']],
        'prompt': [prompt],
        'target': [target],
        'subject' : [rewrite_data['subject']],
        'request' : [rewrite_data['target_new']['str']],
    }
    cur_sum = collections.defaultdict(lambda: [])
    data = record
    # compute ROME metrics
    for prefix in ["pre", "post"]:
        # record essence_drift metric
        if 'essence_score' in data[prefix]:
            cur_sum[f"{prefix}_essence_ppl"] = data[prefix]['essence_score']
            cur_sum[f"{prefix}_essence_prob"] = np.exp(-data[prefix]['essence_score'])
        # Probability metrics for which new should be lower (better) than true
        for key in ["rewrite_prompts_probs", "paraphrase_prompts_probs"]:
            if prefix not in data or key not in data[prefix]:
                continue
            sum_key_discrete = f"{prefix}_{key.split('_')[0]}_success"
            sum_key_cont = f"{prefix}_{key.split('_')[0]}_diff"
            cur_sum[sum_key_discrete].append(
                np.mean(
                    [
                        x["target_true"] > x["target_new"]
                        for x in data[prefix][key]
                    ]
                )
            )
            cur_sum[sum_key_cont].append(
                np.mean(
                    [
                        np.exp(-x["target_new"]) - np.exp(-x["target_true"])
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
                        x["target_true"] < x["target_new"]
                        for x in data[prefix][key]
                    ]
                )
            )
            cur_sum[sum_key_cont].append(
                np.mean(
                    [
                        np.exp(-x["target_true"]) - np.exp(-x["target_new"])
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
    # add post-pre ppl scores
    if 'essence_score' in data['post']:
        cur_sum['essence_ppl_diff'] = cur_sum['post_essence_ppl'] - cur_sum['pre_essence_ppl'] # lower is better
        cur_sum['essence_prob_diff'] = cur_sum['post_essence_prob'] - cur_sum['pre_essence_prob'] # higher is better
    # add ROME metrics to record_dict and append to dataframes
    record_dict.update(cur_sum)
    df = pd.DataFrame(record_dict)
    dataframes.append(df)
  if len(dataframes) > 0:
    return_df = pd.concat(dataframes)
  else:
    return_df = pd.DataFrame()
  return return_df

def main(
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
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Get run hyperparameters
    _model_name = model_name.replace('/', '_')
    params_path = os.path.join(f'{CODE_DIR}/hparams/', alg_name, f"{_model_name}.json")
    if alg_name == 'FT':
      params_path = params_path.replace('.json', '_constr.json')
    hparams = params_class.from_json(params_path)
    if override_hparams is not None:
      hparams.__dict__.update(override_hparams)
    print(f"Executing {alg_name} with parameters {hparams}")

    # Determine run directory
    important_hparam_names = override_hparams.keys() if override_hparams is not None else ['layers']
    important_hparams = {k:v for k,v in hparams.__dict__.items() if any([k==name for name in important_hparam_names])}
    exp_name = ROME_experiment_name(model_name.split('/')[-1],
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
        if rewrite_this_point:
            
            # generate essence_texts for evaluation if needed
            if do_essence_tests or not skip_generation_tests:
                subject = record["requested_rewrite"]['subject']
                essence_prompt = "{} is a".format(subject)
                if len(snips.names_to_samples[subject]) == 0:
                    if verbose:
                        print("GENERATING ESSENCE TEXTS")
                    essence_texts = generate_fast(
                        model,
                        tok,
                        [essence_prompt],
                        n_gen_per_prompt=5,
                        max_out_len=100,
                    )
                    snips.names_to_samples[subject].extend(essence_texts)
                elif verbose:
                    print("using wikipedia essence texts")
            
            # Compute weight changes + record weights that changed
            start = time.time()
            args_conserve_memory = (
                dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
                if conserve_memory
                else dict()
            )
            with torch.enable_grad():
              request = record["requested_rewrite"]
              paraphrase_prompts = record["paraphrase_prompts"]
              neighborhood_prompts = record["neighborhood_prompts"]
              if verbose:
                print(
                    "Updating point:"
                    f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
                    f"\n Paraphrases: {paraphrase_prompts[:2]}"
                    f"\n Neighbors: {neighborhood_prompts[:2]}"
                )
                for text in snips.names_to_samples[request['subject']][:5]:
                    print(f" Essence text: {text[:200]}")
                
              edited_model, weights_copy = apply_algo(
                  model,
                  tok,
                  [request],
                  hparams,
                  copy=False,
                  return_orig_weights=True,
                  **args_conserve_memory,
              )
            exec_time = time.time() - start
            print("Execution took", exec_time)

            # Execute evaluation suite
            start = time.time()
            metrics = {
                "case_id": case_id,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": ds_eval_method(edited_model, tok, record, snips, vec, skip_generation_tests),
            }

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to("cuda")
            metrics["pre"] = ds_eval_method(model, tok, record, snips, vec, skip_generation_tests)

            # print("metrics: ", metrics)
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
    import argparse

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
        "--dataset_size_limit",
        "-n",
        type=int,
        default=1000,
        help="Truncate CounterFact to first n records.",
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
        action="store_true",
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
    parser.set_defaults(skip_generation_tests=True, do_essence_tests=True, conserve_memory=False, verbose=False, overwrite=False)
    args = parser.parse_args()

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
    window_sizes = [1]
    if 'gpt2' in model_name:
        central_layers = list(range(0, 48, 4)) + [17, 47]
        num_layers = 48
    if '6B' in model_name:
        central_layers = list(range(0, 28, 4)) + [5, 27]
        num_layers = 28
    window_sizes=[1]
    if args.edit_layer > -2:
        central_layers = [args.edit_layer]
    print("Starting sweep with hparams:")
    print("- window_sizes: ", window_sizes)
    print("- central_layers: ", central_layers)

    # main experiment loop
    results_dfs = []
    for window_size in window_sizes:
        for central_layer in central_layers:
            override_hparams = get_override_hparams(window_size, central_layer, alg_name)
            if RUN_EXPERIMENT:
                main(
                    alg_name=alg_name,
                    model_name=model_name,
                    ds_name=ds_name,
                    dataset_size_limit=num_points,
                    do_essence_tests=args.do_essence_tests,
                    skip_generation_tests=args.skip_generation_tests,
                    conserve_memory=False,
                    mt=mt,
                    override_hparams=override_hparams,
                    verbose=args.verbose,
                    overwrite=args.overwrite,
                )
            # accumulate reuslts
            exp_name = ROME_experiment_name_from_override_params(model_name, alg_name, ds_name, override_hparams, hparams_class)
            editing_results_df = make_editing_results_df(exp_name, n=num_points)
            editing_results_df['edit_method'] = alg_name
            editing_results_df['edit_central_layer'] = central_layer
            editing_results_df['edit_window_size'] = window_size
            results_dfs.append(editing_results_df)
    
    # combine and save results
    results_df = pd.concat(results_dfs)
    _model_name = model_name.split('/')[-1]
    if len(central_layers) > 1:
        file_name = f'{_model_name}_{alg_name}_outputs_{ds_name}_editing_sweep_n{num_points}.csv'
    else:
        _layer = 'embeds' if central_layers[0] == -1 else central_layers[0]
        print("checking layer naming working properly:")
        print(central_layers)
        print(_layer)
        file_name = f'{_model_name}_{alg_name}_outputs_{ds_name}_editing_layer-{_layer}_n{num_points}.csv'
    save_path = f'{BASE_DIR}/results/{file_name}'
    results_df.to_csv(save_path, index=False)
    # upload results csv to google bucket    
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('research-brain-belief-localization-xgcp')
    blob = bucket.blob(f'output/{file_name}')
    blob.upload_from_filename(save_path)

    print(f"saving csv at {save_path}...")
    print(results_df.loc[:,['case_id', 'subject', 'target', 'request', 'post_rewrite_success', 'post_neighborhood_success', 'post_paraphrase_success', 'post_score', 'essence_ppl_diff', 'essence_prob_diff']])

