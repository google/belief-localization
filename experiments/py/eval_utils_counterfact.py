"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain
from contextlib import nullcontext

import nltk
import numpy as np
import scipy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets
from experiments.causal_trace import layername, corrupted_forward_pass, find_token_range, make_inputs, simple_make_inputs
from util import nethook
from util.generate import generate_fast
from util.perplexity import perplexity

def compute_rewrite_quality_counterfact(
    args,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
    skip_generation_tests: bool,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???

    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]
    attribute_prompts = record["attribute_prompts"]
    generation_prompts = record["generation_prompts"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts,
        attribute_prompts,
    ]
    
    # Flatten all the evaluated prefixes into one list.
    probs = test_batch_prediction(
        args, model, tok, list(chain(*prob_prompts)), target_new["str"], target_true["str"], subject
    )
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
                "attribute_prompts",
            ]
        )
    }
    if snips is not None:
        # Gather reference texts
        rel_id = record["requested_rewrite"]["relation_id"]
        consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]
        # essence_texts = snips.snippets_list
        essence_texts = snips.names_to_samples[subject]
        if len(essence_texts) > 5:
            essence_texts = essence_texts[:5]
        if skip_generation_tests:
            consistency_texts = []
            vec = None
        gen_stats = test_generation(
            args,
            model,
            tok,
            generation_prompts,
            consistency_texts,
            essence_texts,
            vec,
            subject,
        )
        ret.update(gen_stats)

    return ret

def test_batch_prediction(
    args,
    model,
    tok,
    prefixes: typing.List[str],
    target_new: str,
    target_true: str,
    subject: str,
):
    """ """
    
    # calculate the token indices for the subject for each prompt. evaluation gets done in a batch, so need to noise at different token indices depending on the data point
    if args.use_noised_subject:
        prng = np.random.RandomState(1) 
        embed_layername = layername(model, 0, 'embed')
        e_ranges = []
        for prompt in prefixes:
            e_range = find_token_range(tok, substring=subject, prompt_str=prompt)
            e_ranges.append(e_range)
        # define function that noises embeddings at tokens_to_mix indices
        def noise_embeddings(x, layer):
            # corrrupt subject embeddings depending on the datapoint index
            noise_len = e_ranges[0][1] - e_ranges[0][0] # rewrite prompts are first, so they will always include the subject, so safe to index here
            if layer == embed_layername:
                embeds_noise = torch.from_numpy(prng.randn(x.shape[0], noise_len, x.shape[2])).to(x.device)
                for i in range(len(e_ranges)):
                    e_range = e_ranges[i]
                    if e_range is not None:
                        b, e = e_range
                        x[i, b:e] += args.hparams.editing_noise * embeds_noise[i]
                print(f"datapoint {i}: {prefixes[i]}")
                print(f" added noise to embeds at idx {e_ranges[i]}: ", embeds_noise[i])
                return x
            else:
                return x

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with nethook.TraceDict(model, [embed_layername], edit_output=noise_embeddings) if args.use_noised_subject else nullcontext():
        logits = model(**prompt_tok).logits

    results = np.zeros((logits.size(0),), dtype=np.float32)

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            results[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        results[i] /= cur_len

    return [
        {"target_new": results[i].item(), "target_true": results[i + 1].item()}
        for i in range(0, len(results), 2)
    ]


def test_generation(
    args,
    model,
    tok,
    prefixes: typing.List[str],
    consistency_texts: typing.List[str],
    essence_texts: typing.List[str],
    vec: TfidfVectorizer,
    subject: str,
):
    return_dict = {}
    if len(consistency_texts) > 0:
        gen_texts = generate_fast(
            model,
            tok,
            prefixes,
            n_gen_per_prompt=1,
            max_out_len=100,
        )
        return_dict['gen_texts'] = gen_texts

        return_dict['ngram_entropy'] = n_gram_entropy(gen_texts)
        if vec is not None:
            consistency_tfidf = tfidf_similarity(
                " ".join(gen_texts), " ".join(consistency_texts), vec
            )
            return_dict['reference_score'] = consistency_tfidf

    if len(essence_texts) > 0:

        # calculate the token indices for the subject for each ESSENCE TEXT
        if args.use_noised_subject:
            prng = np.random.RandomState(1) 
            embed_layername = layername(model, 0, 'embed')
            e_ranges = []
            for essence_text in essence_texts:
                e_range = find_token_range(tok, substring=subject, prompt_str=essence_text)
                e_ranges.append(e_range)
            # define function that noises embeddings at tokens_to_mix indices
            def noise_embeddings(x, layer):
                # corrrupt subject embeddings depending on the datapoint index
                noise_len = e_ranges[0][1] - e_ranges[0][0]
                if layer == embed_layername:
                    embeds_noise = torch.from_numpy(prng.randn(x.shape[0], noise_len, x.shape[2])).to(x.device)
                    for i in range(len(e_ranges)):
                        e_range = e_ranges[i]
                        if e_range is not None:
                            b, e = e_range
                            x[i, b:e] += args.hparams.editing_noise * embeds_noise[i]
                    print(f"essence text {i}: {essence_texts[i]}")
                    print(f" added noise to embeds at idx {e_ranges[i]}: ", embeds_noise[i])
                    return x
                else:
                    return x

        with nethook.TraceDict(model, [embed_layername], edit_output=noise_embeddings) if args.use_noised_subject else nullcontext():
            ppls = []
            for essence_text in essence_texts:
                ppl = perplexity(model, tok, essence_text, max_input_length=100)
                ppls.append(ppl)
            avg_ppl = np.mean(ppls)
            return_dict.update({"essence_score": avg_ppl, "essence_text": essence_texts})

    return return_dict


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()
