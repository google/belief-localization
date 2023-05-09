import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/zsre_mend_eval.json"


class MENDQADataset:
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """

    def __init__(self, data_dir: str, tok: AutoTokenizer, size: int = None, *args, **kwargs):
        data_dir = Path(data_dir)
        zsre_loc = data_dir / "zsre_mend_eval.json"
        if not zsre_loc.exists():
            print(f"{zsre_loc} does not exist. Downloading from {REMOTE_URL}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, zsre_loc)

        with open(zsre_loc, "r") as f:
            raw = json.load(f)

        # get all possible answers
        all_answers = []
        for i, record in enumerate(raw):
            all_answers.append(record['answers'][0])

        data = []
        print("Loading zsre data...")
        for i, record in enumerate(raw):
            print(f"loading point {i}", end='\r')
            assert (
                "nq question: " in record["loc"]
            ), f"Neighborhood prompt missing `nq question:`. Check for errors?"
            ans_toks = tok(" " + record["loc_ans"])["input_ids"]
            true_answer = record["answers"][0]
            eligible_alt_answers = np.setdiff1d(all_answers, true_answer)
            # adjust prompt formatting a bit -- occurs for paraphrases later too
            question = record["src"].replace(record["subject"], "{}")
            prompt = f"Question: {question} Answer:"
            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": prompt,
                        "subject": record["subject"],
                        # ROME paper uses editing to correct mistakes, unlike the CounterFact setting
                        # "target_new": {"str": record["answers"][0]},
                        # "target_true": {"str": "<|endoftext|>"},
                        # even though error fixing is a better eval, we compatabilize with CounterFact
                        # by setting the new target to a random entity
                        "target_new": {"str": np.random.choice(eligible_alt_answers)},
                        "target_true": {"str": true_answer},  
                    },
                    "paraphrase_prompts": [f"Question: {record['rephrase']} Answer:"],
                    "neighborhood_prompts": [
                        {
                            "prompt": record["loc"] + "?" + tok.decode(ans_toks[:i]),
                            "target": tok.decode(ans_toks[i]),
                        }
                        for i in range(len(ans_toks))
                    ],
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
            )
            if size is not None and i == size:
                break

        self._data = data

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
