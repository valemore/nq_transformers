'''
Implements eval logic for Google's Natural Questions (NQ) task
'''

import collections
import json
import logging
import numpy as np
from pathlib import Path

import nq_config

logger = logging.getLogger(__name__)

RawResult = collections.namedtuple(
    "RawResult",
    ["unique_id", "start_logits", "end_logits", "answer_type_logits"])

def read_candidates(input_path):
    """Read candidates from a single jsonl file."""
    candidates_dict = {}
    logger.info("Reading examples from: %s", input_path)
    input_path = Path(input_path) if isinstance(input_path, str) else input_path
    with open(input_path, "r") as input_file:
        for index, line in enumerate(input_file):
            e = json.loads(line)
            candidates_dict[e["example_id"]] = e["long_answer_candidates"]
    return candidates_dict

def top_k_indices(logits,n_best_size,token_map):
    indices = np.argsort(logits[1:])+1
    indices = indices[token_map[indices]!=-1]
    return indices[-n_best_size:]

class ScoreSummary(object):
    def __init__(self):
        self.predicted_label = None
        self.short_span_score = None
        self.cls_token_score = None
        self.answer_type_logits = None

Span = collections.namedtuple("Span", ["start_token_idx", "end_token_idx"])

def compute_predictions(example):
    """Converts an example into an NQEval object for evaluation."""
    predictions = []
    n_best_size = nq_config.n_best_size
    max_answer_length = nq_config.max_answer_length
    i = 0
    for unique_id, result in example.results.items():
        if unique_id not in example.features:
            raise ValueError("No feature found with unique_id:", unique_id)
        token_map = np.array(example.features[unique_id]["token_map"]) #.int64_list.value
        start_indexes = top_k_indices(result['start_logits'],n_best_size,token_map)
        if len(start_indexes)==0:
            continue
        end_indexes   = top_k_indices(result['end_logits'],n_best_size,token_map)
        if len(end_indexes)==0:
            continue
        indexes = np.array(list(np.broadcast(start_indexes[None],end_indexes[:,None])))  
        indexes = indexes[(indexes[:,0]<indexes[:,1])*(indexes[:,1]-indexes[:,0]<max_answer_length)]
        for start_index,end_index in indexes:
            summary = ScoreSummary()
            summary.short_span_score = (
                result['start_logits'][start_index] +
                result['end_logits'][end_index])
            summary.cls_token_score = (
                result['start_logits'][0] + result['end_logits'][0])
            answer_type_logits = np.array(result['answer_type_logits'])
            summary.answer_type_logits = (answer_type_logits - answer_type_logits.mean())
            start_span = token_map[start_index]
            end_span = token_map[end_index] + 1

            # Span logits minus the cls logits seems to be close to the best.
            score = summary.short_span_score - summary.cls_token_score
            predictions.append((score, i, summary, start_span, end_span))
            i += 1 # to break ties

    # Default empty prediction.
    score = -10000.0
    short_span = Span(-1, -1)
    long_span  = Span(-1, -1)
    summary    = ScoreSummary()

    if predictions:
        score, _, summary, start_span, end_span = sorted(predictions, reverse=True)[0]
        short_span = Span(start_span, end_span)
        for c in example.candidates:
            start = short_span.start_token_idx
            end = short_span.end_token_idx
            if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:
                long_span = Span(c["start_token"], c["end_token"])
                break

    summary.predicted_label = {
        "example_id": int(example.example_id),
        "long_answer": {
            "start_token": int(long_span.start_token_idx),
            "end_token": int(long_span.end_token_idx),
            "start_byte": -1,
            "end_byte": -1
        },
        "long_answer_score": float(score),
        "short_answers": [{
            "start_token": int(short_span.start_token_idx),
            "end_token": int(short_span.end_token_idx),
            "start_byte": -1,
            "end_byte": -1
        }],
        "short_answer_score": float(score),
        "yes_no_answer": "NONE",
        "answer_type_logits": summary.answer_type_logits.tolist(),
        "answer_type": int(np.argmax(summary.answer_type_logits))
    }

    return summary

class EvalExample(object):
    """Eval data available for a single example."""
    def __init__(self, example_id, candidates):
        self.example_id = example_id
        self.candidates = candidates
        self.results = {}
        self.features = {}

def compute_pred_dict(candidates_dict, dev_features, raw_results, tqdm=None):
    """Computes official answer key from raw logits."""
    raw_results_by_id = [(int(res['unique_id']),1, res) for res in raw_results]

    examples_by_id = [(int(k), 0, v) for k, v in candidates_dict.items()]
  
    features_by_id = [(int(dev_features[i]['unique_id']), 2, dev_features[i]) for i in range(len(dev_features))]

    examples = []
    merged = sorted(examples_by_id + raw_results_by_id + features_by_id)
    for idx, type_, datum in merged:
        if type_ == 0:
            examples.append(EvalExample(idx, datum))
        elif type_ == 2:
            examples[-1].features[idx] = datum
        else:
            examples[-1].results[idx] = datum

    nq_pred_dict = {}
    if tqdm is not None:
        examples = tqdm(examples)
    for e in examples:
        summary = compute_predictions(e)
        nq_pred_dict[e.example_id] = summary.predicted_label

    return nq_pred_dict
