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
    KnownsDataset,
    get_tfidf_vectorizer,
)
from experiments.causal_trace import ModelAndTokenizer, score_from_batch, get_high_and_low_scores, plot_trace_heatmap
from experiments.causal_trace import calculate_hidden_flow, layername, corrupted_forward_pass, find_token_range, make_inputs, simple_make_inputs, predict_model
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.generate import generate_fast
from util.globals import *
from util.fewshot_utils import first_appearance_fewshot_accuracy_sum, fewshot_accuracy_sum


# globals
CODE_DIR='/home/peterhase/belief-loc'
BASE_DIR='/home/peterhase'

# functions
def load_counterfact_dataset(args):
    counterfacts = CounterFactDataset(DATA_DIR)
    generate_completions = False
    knowledge_inputs = []
    knowledge_targets = []
    gpt_completions = []
    subjects = []
    if args.verbose:
        print('\n')
    for id, record in enumerate(counterfacts):
        if args.verbose:
            print("starting record: ", id, end='\r')
        rewrite_data = record['requested_rewrite']
        prompt = rewrite_data['prompt'].format(rewrite_data['subject'])
        target = rewrite_data['target_true']['str']
    if generate_completions:
        completion, _, _ = predict_model(mt, [prompt], max_decode_steps=10)
        completion = completion[0]
        print(prompt, " -- ", completion)
    else:
        completion = ''
    knowledge_inputs.append(prompt)
    knowledge_targets.append(target)
    gpt_completions.append(completion)
    subjects.append(rewrite_data['subject'])

    counterfact_data = pd.DataFrame({'input': knowledge_inputs, 
                                'label_str': knowledge_targets,
                                'completion': gpt_completions,
                                'subject' : subjects,
                                })
    return counterfact_data

def format_time(runtime):
  if runtime > 3600:
    return f"{runtime/3600:.2f} hours"
  if runtime > 60:
    return f"{runtime/60:.2f} minutes"
  else:
    return f"{runtime:.2f} secs"

def load_factual_dataset(args):
    knowns = KnownsDataset(DATA_DIR)
    knowledge_inputs = [knowledge['prompt'].strip() for knowledge in knowns]
    knowledge_targets = [knowledge['attribute'] for knowledge in knowns]
    gpt_completions = [knowledge['prediction'] for knowledge in knowns]
    factual_data = pd.DataFrame({'input': knowledge_inputs, 
                                'label_str': knowledge_targets,
                                'completion': gpt_completions,
                                'subject' : [knowledge['subject'] for knowledge in knowns]
                                })
    return factual_data

def pull_prompt_from_data(data, k):
  prompt_idx = np.random.choice(np.arange(len(data)), size=k, replace=False)
  prompt_ex = data.iloc[prompt_idx]
  eval_idx = np.setdiff1d(np.arange(len(data)), prompt_idx)
  eval_data = data.iloc[eval_idx]
  return prompt_ex, eval_data

def verbalize(label, answers, inverted_labels=False):
  '''
  maps integer labels to string answers for scoring by LM
  '''
  assert label < len(answers), f"requesting label {label} but only {len(answers)} answers"
  if not inverted_labels:
    return answers[label]
  else:
    assert len(answers) == 2, "using inverted_labels=True but more than two answers provided"
    return answers[1-label]

def format_example(input, label_str="", answers=None, cot_reason=None, template_id=0):
  if template_id == 0:
    return f"\"{input}\" is {label_str}"
  if template_id == 1:
    return f"The scenario \"{input}\" describes an action that is {label_str}"
  if template_id == 2:
    return f"scenario: \"{input}\" label ({answers[0]}/{answers[1]}): {label_str}"
  # Hendrycks cm format. label is "wrong" or "not wrong". used with k=32
  if template_id == 3: 
    if label_str=="": # assume test input, don't add line breaks at end
      return "{}\nReaction: this is {}".format(input, label_str)
    else:
      return "{}\nReaction: this is {}\n".format(input, label_str)
  if template_id == 4:
    if label_str=="": # assume test input
      return f"\"{input}\""
    else:
      return f"\"{input}\" {cot_reason} Therefore, the action is {label_str}"
  if template_id == 5:
    if label_str=="": # assume test input
      return f"\"{input}\" The action is"
    else:
      return f"\"{input}\" The action is {label_str} because {cot_reason}"
  # control condition for CoT above, but for multiple choice
  if template_id == 6:
    if cot_reason is not None:
      return_str = f"\"{input}\" {cot_reason} Therefore, the action is"
    else:
      return_str = f"\"{input}\" Therefore, the action is"
    if label_str != "": # not a test input
      return_str += f" {label_str}"
    return return_str
  # used with chain of thought reasons that re-specify the action
  if template_id == 7:
    if label_str=="": # assume test input
      return f"\"{input}\""
    else:
      return f"\"{input}\" {cot_reason} {label_str}"
  if template_id == 8: # for factual data completions
    if label_str=="": # assume test input
      return f"{input}"
    else:
      return f"{input} {label_str}"
  else:
    raise ValueError(f"Not implemented template for template_id {template_id}")

def format_prompt(examples, test_input, instructions=None, separator='\n'):
  # takes list of examples, test_input, already processed by format_example
  if len(examples) > 0:
    examples = separator.join(examples)
    prompt = examples + separator + test_input
  else:
    prompt = test_input
  if instructions:
    prompt = instructions + separator + prompt
  return prompt

def format_example_from_df_row(df_row, template_id=0):
  input = df_row.input
  label_str = df_row.label_str
  example = format_example(input, label_str, template_id=template_id)
  return example

def format_prompt_from_df(df, test_input, answers=None, instructions=None, cot_reasons=None, separator='\n', template_id=0, idx=None):
  # read data from df and pass to format_prompt()
  # add chain-of-thought reasons via format_example here
  examples = []
  select_df = df.iloc[idx,:] if idx else df
  for data_num, (_, df_row) in enumerate(select_df.iterrows()):
    input = df_row['input']
    label_str = df_row['label_str']
    cot_reason = cot_reasons[data_num] if cot_reasons else None
    example = format_example(input, label_str, answers=answers, cot_reason=cot_reason, template_id=template_id)
    examples.append(example)
  formatted_test_input = format_example(test_input, template_id=template_id)
  prompt = format_prompt(examples, formatted_test_input, instructions=instructions, separator=separator)
  return prompt

def make_results_df(model_name, exp_name, count=1208):
  all_data_points = []
  print(f"Making results_df for exp: {exp_name}...")
  for kind in [None, 'mlp', 'attn']:
    read_count = 0
    for data_point_id in range(count):
      path = f"{BASE_DIR}/results/{model_name}/traces/{exp_name}_{data_point_id}_{kind}.csv"
      if os.path.exists(path):
        data = pd.read_csv(path)
        read_count += 1
        all_data_points.append(data)
      else:
        print(f"skipping point {path}")
  results_df = pd.concat([result_df for result_df in all_data_points])  
  return results_df

def results_dict_to_df(results_dict, tokenizer, exp_name, task_name, split_name):
  # format result_dict as pandas df for saving as csv
  # saving in 'long' format, with one row per restored token and layer_idx (i.e. per unique prob_effect)
  num_tokens = len(results_dict['test_input_tokens'])
  subj_begin_idx, subj_end_idx_plus_one = results_dict['subject_range']
  num_restored_tokens, num_layers = results_dict['scores'].shape
  df_dicts = []
  for token_idx in range(num_tokens):
    token_str = results_dict['test_input_tokens'][token_idx]
    for layer_idx in range(num_layers):
      if token_idx < subj_begin_idx:
        restore_prob = results_dict['low_score']
        is_subj_token = False
      else:
        restore_prob = results_dict['scores'][token_idx-subj_begin_idx, layer_idx].item()
        is_subj_token = True
      df_dict = {
          'input_id': results_dict['input_id'],
          'experiment_name' : exp_name,
          'task' : task_name,
          'split' : split_name,
          'input_str' : results_dict['test_input_str'],
          'label_str' : results_dict['label_str'],
          'pred_str' : results_dict['answer'],
          'orig_pred_prob' : results_dict['high_score'],
          'corrupted_pred_prob' : results_dict['low_score'],
          'subj_begin_idx' : results_dict['subject_range'][0],
          'subj_end_idx' : results_dict['subject_range'][1],
          'token_idx' : token_idx,
          'layer_idx' : layer_idx,
          'restore_prob' : restore_prob,
          'token_str' : token_str,
          'module' : results_dict['kind'] if results_dict['kind'] else 'None',
          'seq_len' : num_tokens,
          'last_seq_token' : token_idx == num_tokens - 1,
          'is_correct': results_dict['correct_prediction'],
          'is_subj_token' : is_subj_token,
      }
      df_dict = {k: [v] for k,v in df_dict.items()}
      df_dicts.append(df_dict)
  df = pd.concat([pd.DataFrame(point) for point in df_dicts])
  return df

def causal_tracing_loop(experiment_name, task_name, split_name, mt, eval_data, 
                        num_samples, noise_sd, restore_module, window_size, show_plots, 
                        explain_quantity,
                        k, random_seed=0, n=None, prompt_data=None, 
                        instructions=None, answers=None, template_id=0, cot_reasons=None,
                        max_decode_steps=128, extract_answers=None,
                        trigger_phrase=None, print_examples=0, save_plots=True,
                        overwrite=False, 
                        correctness_filter=False,
                        check_corruption_effects=False,
                        min_corruption_effect = 0,
                        min_pred_prob=0):
  """Runs causal tracing algorithm over a dataset provided in eval_data.
  args:
    explain_quantity: in ['label', 'score_pred', None], we explain p(explain_quantity)
      None means that you generate a prediction, 'score_pred' means you score to get pred
    check_corruption_effects: instead of doing causal tracing, loop over the data and check
      the effect of the subject noising step on the output. used for calibrating the noise size
  """
  # eval model and return a single row df with the results
  start = time.time()
  print(f"Causal tracing for experiment: {experiment_name}...")
  # argument checks
  if k > 0 and prompt_data is None: 
    assert len(prompt_data) == k, f"need to provide prompt data of len {k}"
  if prompt_data is None:
    prompt_data = pd.data.frame({'x':[]})
  if answers and not extract_answers:
    extract_answers = answers
  # subsample eval data if requested. TAKE FIRST n SAMPLES
  if n is not None:
    eval_data_loop = eval_data[:n] 
    # eval_data_loop = eval_data.sample(n=n, random_state=random_seed, replace=False)
  else:
    eval_data_loop = eval_data
  # begin eval loop
  effective_batch_size = 1
  n_chunks = np.ceil(len(eval_data_loop) / effective_batch_size)
  causal_tracing_results = []
  skipped = 0
  for batch_num, batch in enumerate(np.array_split(eval_data_loop, n_chunks)):
    data_point_id = batch.index[0]
    # format data
    input = batch.input.item()
    if task_name in ['commonsense', 'utilitarianism', 'deontology', 'justice', 'virtue']:
      subject = input
    elif 'fact' in task_name:
      subject = batch.subject.item()
    label = batch.label_str.item()
    query_input = format_prompt_from_df(prompt_data, 
                                      input, 
                                      answers=answers, 
                                      instructions=instructions, 
                                      cot_reasons=cot_reasons, 
                                      separator='\n', 
                                      template_id=template_id)
    # get model is_correct variable
    with torch.no_grad():
      preds, scores, query_inputs = predict_model(mt, 
                                                  [query_input], 
                                                  answers, 
                                                  trigger_phrase=trigger_phrase, 
                                                  max_decode_steps=max_decode_steps)
      # record stats
      # first case is when we are generating predictions and extracting answers from them
      if answers is None and extract_answers is not None:
        is_correct = first_appearance_fewshot_accuracy_sum(preds, [label], 
                                                           extract_answers=extract_answers, 
                                                           trigger_phrase=trigger_phrase)
      else:
        is_correct = fewshot_accuracy_sum(preds, [label])
      if correctness_filter is True:
        if not is_correct:
          print(f"skipping batch {batch_num}, point {data_point_id}, as it is wrongly predicted")
          continue
    # get tracing output to explain
    if explain_quantity == 'label':
      tracing_target = label
    elif explain_quantity == 'score_pred':
      tracing_target = preds[0]
    else:
      tracing_target = None

    # start causal tracing loop
    if print_examples > 0 and batch_num <= print_examples:
      printing=True
    else:
      printing=False
    time_per_point = (time.time()-start) / (batch_num-skipped) if (batch_num-skipped) > 0 else -1
    print(f"Point {batch_num}, id {data_point_id}, time/point: {format_time(time_per_point)}")
    if printing:
      print("Full query:\n", query_input)
      print("subject to noise: ", subject)
      print("target tokens: ", label)
      print("tracing output to be explained: ", tracing_target)
      print("pred: ", preds)
      print("correct: ", is_correct)

    # check_corruption_effects means we 
    if check_corruption_effects:
      high_score, low_score = get_high_and_low_scores(
        mt, query_input, subject, target=tracing_target, samples=num_samples, noise=noise_sd, 
      )
      diff = high_score-low_score
      print(f"high score: {high_score:.2f}, low_score: {low_score:.2f}, diff: {diff:.2f}\n")
      if min_pred_prob > 0:
        if high_score < min_pred_prob:
          print(f"skipping batch {batch_num}, point {data_point_id}, with too small a pred prob of {high_score:.3f}")
          continue
      if min_corruption_effect > 0:
        if diff < min_corruption_effect:
          print(f"skipping batch {batch_num}, point {data_point_id}, with too small a corruption effect of {diff:.3f}")
          continue
      else:
        continue

    kinds = [restore_module] if restore_module!=None else [None, "mlp", "attn"]
    for kind in kinds:
      # potentially skip if exists
      if not overwrite:
        _model_name = model_name.split('/')[-1]
        save_path = f"{BASE_DIR}/results/{_model_name}/traces/{experiment_name}_{data_point_id}_{kind}.csv"
        if os.path.exists(save_path):
          if printing:
            print(f"skipping batch {batch_num}, point {data_point_id}, as it is already written")
          skipped += 1
          continue
      if printing: 
        print("starting module: ", kind)
      
      # CALCULUATE HIDDEN FLOW
      results_dict = calculate_hidden_flow(
        mt, query_input, subject, target=tracing_target, samples=num_samples, noise=noise_sd, window=window_size, kind=kind,
      )
      # add variables to results_dict
      results_dict['input_id'] = data_point_id
      results_dict['label_str'] = label
      results_dict['correct_prediction'] = is_correct
      results_df = results_dict_to_df(results_dict, mt.tokenizer, experiment_name, task_name, split_name)
      causal_tracing_results.append(results_df)
      # plot and save results (both results_dict, for their plotting code, and the results_df, for ours)
      if save_plots:
        plot_name = f"{experiment_name}_plot{data_point_id}_{kind}.pdf"
        save_path = os.path.join(f'{BASE_DIR}/results/{_model_name}/traces', plot_name) if plot_name else None 
        print(f"saving plot at {save_path}")
        _model_name = model_name.split('/')[-1]
        plot_trace_heatmap(results_dict, show_plot=show_plots, savepdf=save_path, modelname=_model_name)
        save_path = f"{BASE_DIR}/results/{_model_name}/traces/{experiment_name}_{data_point_id}_{kind}.npz"
        if printing:
          print(f"saving results at {save_path}")
        np.savez(save_path, results_dict)
        results_df.to_csv(save_path.replace('npz', 'csv'), index=False)
    del batch, input, label, subject, query_input
  # make results dfs
  if len(causal_tracing_results) > 0:
    results_df = pd.concat([result_df for result_df in causal_tracing_results])
  else:
    results_df = None
  full_prompt = format_prompt_from_df(prompt_data, "{test_input}", answers=answers, instructions=instructions, cot_reasons=cot_reasons, separator='\n', template_id=template_id)
  metadata_df = pd.DataFrame({
      'exp_name': [exp_name],
      'task_name': [task_name],
      'k': [k],
      'cot' : [cot_reasons is not None],
      'exact_prompt': [full_prompt]
  })
  # make metadata for df
  print("Done! Runtime: ", format_time(time.time()-start))
  return results_df, metadata_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["counterfact", "zsre"],
        default="counterfact",
        help="Dataset to perform evaluations on. Either CounterFact (cf) or zsRE (zsre).",
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
        "--overwrite",
        action="store_true",
        help="Overwrite previous experiment results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="More printing",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=1,
        choices=[0,1],
    )
    parser.set_defaults(skip_generation_tests=True, do_essence_tests=True, conserve_memory=True, verbose=False, overwrite=False,
                        use_noised_target=False, use_noised_subject=False)
    args = parser.parse_args()

    # set seed
    RANDOM_SEED=1
    np.random.seed(RANDOM_SEED)

    # run experiment
    if args.run:
        torch.set_grad_enabled(False)

        model_name = args.model_name
        
        torch_dtype = torch.float16 if '20b' in model_name else None
        mem_usage = True

        if '20b' not in model_name:
            mt = ModelAndTokenizer(model_name, low_cpu_mem_usage=mem_usage, torch_dtype=torch_dtype)
            torch.cuda.empty_cache()
            mt.model.eval().cuda()
            mt.tokenizer.add_special_tokens({'pad_token' : mt.tokenizer.eos_token})
        else:
            from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
            model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", 
                                                        device_map={
                                                            'embed_out' : 0,
                                                            'gpt_neox.embed_in' : 0,
                                                            'gpt_neox.layers': 1,
                                                            'gpt_neox.final_layer_norm' : 0,
                                                        },
                                                        low_cpu_mem_usage=False,
                                                        torch_dtype=torch_dtype)
            torch.cuda.empty_cache()
            model.eval().cuda()
            tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
            mt = ModelAndTokenizer(model=model, tokenizer=tokenizer, torch_dtype=torch_dtype)

        _model_name = os.path.split(model_name)[-1]
        os.makedirs(f'{BASE_DIR}/results/{_model_name}', exist_ok=True)    
        os.makedirs(f'{BASE_DIR}/results/{_model_name}/traces', exist_ok=True)    
        
    # begin tracing
    template_id = 8
    k = 0
    restore_module = None
    ovr_exp_name = f"{_model_name}_{args.ds_name}_k{k}_sd{RANDOM_SEED}_tracing_sweep_n{args.dataset_size_limit}"
    print("Starting experiment: ", ovr_exp_name)

    if args.ds_name == 'counterfact':
        use_data = load_counterfact_dataset(args)
    if args.ds_name == 'factual':
        use_data = load_factual_dataset(args)
    prompt_ex, eval_data = pull_prompt_from_data(use_data, k)

    # trace args
    num_samples = 10
    window_sizes = [int(x) for x in args.window_sizes.split()]
    if 'gpt2-xl' in args.model_name:
        noise_sd = .1
        max_decode_steps=36
    elif 'gpt-j-6B' in args.model_name:
        # they use .025 (use to recreate orig plots), though it seems like 3*sd is .094, and 3*sd is a rule they use elsewhere.
        noise_sd = .094
        max_decode_steps=36
    elif 'neox' in args.model_name:
        noise_sd = .03
        max_decode_steps=24
    else:
        noise_sd = .01
        max_decode_steps=36

    results_dfs = []
    for window_size in window_sizes:
        _model_name = model_name.split('/')[-1]
        exp_name = f"{_model_name}_{args.ds_name}_k{k}_wd{window_size}_sd{RANDOM_SEED}"
        if args.run:
            results_df, metadata_df = causal_tracing_loop(exp_name, args.ds_name, "", mt, eval_data,
                                        num_samples, noise_sd, restore_module, window_size, 
                                        max_decode_steps=max_decode_steps,
                                        explain_quantity='label',
                                        show_plots=False, 
                                        save_plots=True,
                                        k=k, 
                                        answers=None,
                                        n=args.dataset_size_limit, 
                                        random_seed=RANDOM_SEED, 
                                        prompt_data=prompt_ex,
                                        template_id=template_id, 
                                        print_examples=10,
                                        overwrite=False)
        results_df = make_results_df(_model_name, exp_name, count=args.dataset_size_limit)
        results_df['trace_window_size'] = window_size
        results_dfs.append(results_df)

    all_results_df = pd.concat(results_dfs)
    save_path = f'{BASE_DIR}/results/{ovr_exp_name}.csv'
    results_df.to_csv(save_path, index=False)
    # upload results csv to google bucket
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('research-brain-belief-localization-xgcp')
    blob = bucket.blob(f'output/{ovr_exp_name}.csv')
    blob.upload_from_filename(save_path)
