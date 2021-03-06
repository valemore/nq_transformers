'''
Feature computation for Google's Natural Questions (NQ) task
'''

import collections
import enum
import h5py
import json
import logging
import numpy as np
from pathlib import Path
import re
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import google_tokenization as tokenization

import nq_config

logger = logging.getLogger(__name__)

TextSpan = collections.namedtuple("TextSpan", "token_positions text")

class AnswerType(enum.IntEnum):
    """Type of NQ answer."""
    UNKNOWN = 0
    YES = 1
    NO = 2
    SHORT = 3
    LONG = 4

class Answer(collections.namedtuple("Answer", ["type", "text", "offset"])):
    """Answer record.

    An Answer contains the type of the answer and possibly the text (for
    long) as well as the offset (for extractive).
    """

    def __new__(cls, type_, text=None, offset=None):
        return super(Answer, cls).__new__(cls, type_, text, offset)

class NqExample(object):
    """A single training/test example."""

    def __init__(self,
                 example_id,
                 qas_id,
                 questions,
                 doc_tokens,
                 doc_tokens_map=None,
                 answer=None,
                 start_position=None,
                 end_position=None):
        self.example_id = example_id
        self.qas_id = qas_id
        self.questions = questions
        self.doc_tokens = doc_tokens
        self.doc_tokens_map = doc_tokens_map
        self.answer = answer
        self.start_position = start_position
        self.end_position = end_position

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 answer_text="",
                 answer_type=AnswerType.SHORT):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.answer_text = answer_text
        self.answer_type = answer_type

def get_text_span(example, span):
    """Returns the text in the example's document in the given token span."""
    token_positions = []
    tokens = []
    for i in range(span["start_token"], span["end_token"]):
        t = example["document_tokens"][i]
        if not t["html_token"]:
            token_positions.append(i)
            token = t["token"].replace(" ", "")
            tokens.append(token)
    return TextSpan(token_positions, " ".join(tokens))

def get_candidate_text(e, idx):
    """Returns a text representation of the candidate at the given index."""
    # No candidate at this index.
    if idx < 0 or idx >= len(e["long_answer_candidates"]):
        return TextSpan([], "")

    # This returns an actual candidate.
    return get_text_span(e, e["long_answer_candidates"][idx])

def should_skip_context(e, idx):
    if (nq_config.skip_nested_contexts and
            not e["long_answer_candidates"][idx]["top_level"]):
        return True
    elif not get_candidate_text(e, idx).text.strip():
        # Skip empty contexts.
        return True
    else:
        return False

def candidates_iter(e):
    """Yield's the candidates that should not be skipped in an example."""
    for idx, c in enumerate(e["long_answer_candidates"]):
        if should_skip_context(e, idx):
            continue
        yield idx, c

def get_candidate_type(e, idx):
    """Returns the candidate's type: Table, Paragraph, List or Other."""
    c = e["long_answer_candidates"][idx]
    first_token = e["document_tokens"][c["start_token"]]["token"]
    if first_token == "<Table>":
        return "Table"
    elif first_token == "<P>":
        return "Paragraph"
    elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
        return "List"
    elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
        return "Other"
    else:
        logger.warning("Unknown candidate type found: %s", first_token)
        return "Other"

def add_candidate_types_and_positions(e):
    """Adds type and position info to each candidate in the document."""
    counts = collections.defaultdict(int)
    for idx, c in candidates_iter(e):
        context_type = get_candidate_type(e, idx)
        if counts[context_type] < nq_config.max_position:
            counts[context_type] += 1
        c["type_and_position"] = "[%s=%d]" % (context_type, counts[context_type])

def has_long_answer(a):
    return (a["long_answer"]["start_token"] >= 0 and
                a["long_answer"]["end_token"] >= 0)

def token_to_char_offset(e, candidate_idx, token_idx):
    """Converts a token index to the char offset within the candidate."""
    c = e["long_answer_candidates"][candidate_idx]
    char_offset = 0
    for i in range(c["start_token"], token_idx):
        t = e["document_tokens"][i]
        if not t["html_token"]:
            token = t["token"].replace(" ", "")
            char_offset += len(token) + 1 # + 1 is for the space
    return char_offset

def get_first_annotation(e):
    """Returns the first short or long answer in the example.

    Args:
        e: (dict) annotated example.

    Returns:
        annotation: (dict) selected annotation
        annotated_idx: (int) index of the first annotated candidate.
        annotated_sa: (tuple) char offset of the start and end token
                of the short answer. The end token is exclusive.
    """

    if "annotations" not in e:
        return None, -1, (-1, -1)

    positive_annotations = sorted(
        [a for a in e["annotations"] if has_long_answer(a)],
        key=lambda a: a["long_answer"]["candidate_index"])

    for a in positive_annotations:
        if a["short_answers"]:
            idx = a["long_answer"]["candidate_index"]
            start_token = a["short_answers"][0]["start_token"]
            end_token = a["short_answers"][-1]["end_token"] # QUESTION: WHY LAST SHORT ANSWER?
            return a, idx, (token_to_char_offset(e, idx, start_token),
                            token_to_char_offset(e, idx, end_token) - 1) # QUESTION: WHY  -1 ? ANSWER: end should be exclusive

    for a in positive_annotations:
        idx = a["long_answer"]["candidate_index"]
        return a, idx, (-1, -1)

    return None, -1, (-1, -1)

def get_candidate_type_and_position(e, idx):
    """Returns type and position info for the candidate at the given index."""
    if idx == -1:
        return "[NoLongAnswer]"
    else:
        return e["long_answer_candidates"][idx]["type_and_position"]

def create_example_from_jsonl(line):
    """Creates an NQ example from a given line of JSON."""
    e = json.loads(line, object_pairs_hook=collections.OrderedDict)
    document_tokens = e["document_text"].split(" ")
    e["document_tokens"] = []
    for token in document_tokens:
        e["document_tokens"].append({"token":token, "start_byte":-1, "end_byte":-1, "html_token":"<" in token})

    add_candidate_types_and_positions(e)
    annotation, annotated_idx, annotated_sa = get_first_annotation(e)

    # annotated_idx: index of the first annotated context, -1 if null.
    # annotated_sa: short answer start and end char offsets, (-1, -1) if null.
    question = {"input_text": e["question_text"]}
    answer = {
        "candidate_id": annotated_idx,
        "span_text": "",
        "span_start": -1,
        "span_end": -1,
        "input_text": "long",
    }

    # Yes/no answers are added in the input text.
    if annotation is not None:
        assert annotation["yes_no_answer"] in ("YES", "NO", "NONE")
        if annotation["yes_no_answer"] in ("YES", "NO"):
            answer["input_text"] = annotation["yes_no_answer"].lower()

    # Add a short answer if one was found.
    if annotated_sa != (-1, -1):
        answer["input_text"] = "short"
        span_text = get_candidate_text(e, annotated_idx).text
        answer["span_text"] = span_text[annotated_sa[0]:annotated_sa[1]]
        answer["span_start"] = annotated_sa[0]
        answer["span_end"] = annotated_sa[1]
        expected_answer_text = get_text_span(
            e, {
                "start_token": annotation["short_answers"][0]["start_token"],
                "end_token": annotation["short_answers"][-1]["end_token"],
            }).text
        assert expected_answer_text == answer["span_text"], (expected_answer_text,
                                                                answer["span_text"])

    # Add a long answer if one was found.
    elif annotation and annotation["long_answer"]["candidate_index"] >= 0:
        answer["span_text"] = get_candidate_text(e, annotated_idx).text
        answer["span_start"] = 0
        answer["span_end"] = len(answer["span_text"])

    context_idxs = [-1]
    context_list = [{"id": -1, "type": get_candidate_type_and_position(e, -1)}]
    context_list[-1]["text_map"], context_list[-1]["text"] = (get_candidate_text(e, -1))
    for idx, _ in candidates_iter(e):
        context = {"id": idx, "type": get_candidate_type_and_position(e, idx)}
        context["text_map"], context["text"] = get_candidate_text(e, idx)
        context_idxs.append(idx)
        context_list.append(context)
        if len(context_list) >= nq_config.max_windows:
            break

    if "document_title" not in e:
        e["document_title"] = e["example_id"]

    # Assemble example.
    example = {
        "name": e["document_title"],
        "id": str(e["example_id"]),
        "questions": [question],
        "answers": [answer],
        "has_correct_context": annotated_idx in context_idxs
    }

    single_map = []
    single_context = []
    offset = 0
    for context in context_list:
        single_map.extend([-1, -1]) # QUESTION: ???
        single_context.append("[ContextId=%d] %s" % (context["id"], context["type"]))
        offset += len(single_context[-1]) + 1 # QUESTION: Why + 1?
        if context["id"] == annotated_idx:
            answer["span_start"] += offset
            answer["span_end"] += offset

        # Many contexts are empty once the HTML tags have been stripped, so we
        # want to skip those.
        if context["text"]:
            single_map.extend(context["text_map"])
            single_context.append(context["text"])
            offset += len(single_context[-1]) + 1 # QUESTION: Why + 1?

    example["contexts"] = " ".join(single_context)
    example["contexts_map"] = single_map
    if annotated_idx in context_idxs:
        expected = example["contexts"][answer["span_start"]:answer["span_end"]]

        # This is a sanity check to ensure that the calculated start and end
        # indices match the reported span text. If this assert fails, it is likely
        # a bug in the data preparation code above.
        assert expected == answer["span_text"], (expected, answer["span_text"])

    return example

def make_nq_answer(contexts, answer):
    """Makes an Answer object following NQ conventions.

    Args:
        contexts: string containing the context
        answer: dictionary with `span_start` and `input_text` fields

    Returns:
        an Answer object. If the Answer type is YES or NO or LONG, the text
        of the answer is the long answer. If the answer type is UNKNOWN, the text of
        the answer is empty.
    """
    start = answer["span_start"]
    end = answer["span_end"]
    input_text = answer["input_text"]

    if (answer["candidate_id"] == -1 or start >= len(contexts) or
            end > len(contexts)):
        answer_type = AnswerType.UNKNOWN
        start = 0
        end = 1
    elif input_text.lower() == "yes":
        answer_type = AnswerType.YES
    elif input_text.lower() == "no":
        answer_type = AnswerType.NO
    elif input_text.lower() == "long":
        answer_type = AnswerType.LONG
    else:
        answer_type = AnswerType.SHORT

    return Answer(answer_type, text=contexts[start:end], offset=start)

def read_nq_entry(entry, dataset_type):
    """Converts a NQ entry into a list of NqExamples."""

    def is_whitespace(c):
        return c in " \t\r\n" or ord(c) == 0x202F

    examples = []
    contexts_id = entry["id"]
    contexts = entry["contexts"]
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in contexts:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    questions = []
    for i, question in enumerate(entry["questions"]):
        qas_id = "{}".format(contexts_id)
        question_text = question["input_text"]
        start_position = None
        end_position = None
        answer = None
        if dataset_type == 'train':
            answer_dict = entry["answers"][i]
            answer = make_nq_answer(contexts, answer_dict)

            # For now, only handle extractive, yes, and no.
            if answer is None or answer.offset is None:
                continue
            start_position = char_to_word_offset[answer.offset]
            end_position = char_to_word_offset[answer.offset + len(answer.text) - 1]

            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(
                tokenization.whitespace_tokenize(answer.text))
            if actual_text.find(cleaned_answer_text) == -1:
                logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                continue

        questions.append(question_text)
        example = NqExample(example_id=int(contexts_id),
                            qas_id=qas_id,
                            questions=questions[:],
                            doc_tokens=doc_tokens,
                            doc_tokens_map=entry.get("contexts_map", None),
                            answer=answer,
                            start_position=start_position,
                            end_position=end_position)
        examples.append(example)
    return examples

def read_nq_examples(input_file, dataset_type):
    """Read a NQ json file into a list of NqExample."""
    input_data = []

    logger.info("Reading: %s", input_file)
    with open(input_file, 'r') as input_file:
        for line in input_file:
            input_data.append(create_example_from_jsonl(line))
  
    examples = []
    for entry in input_data:
        examples.extend(read_nq_entry(entry, dataset_type))
    return len(examples), examples

def check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

# A special token in NQ is made of non-space chars enclosed in square brackets.
_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)

def tokenize(tokenizer, text, apply_basic_tokenization=False):
    """Tokenizes text, optionally looking up special tokens separately.

    Args:
        tokenizer: a tokenizer from bert.tokenization.FullTokenizer
        text: text to tokenize
        apply_basic_tokenization: If True, apply the basic tokenization. If False,
            apply the full tokenization (basic + wordpiece).

    Returns:
        tokenized text.

    A special token is any text with no spaces enclosed in square brackets with no
    space, so we separate those out and look them up in the dictionary before
    doing actual tokenization.
    """
    tokenize_fn = tokenizer.tokenize
    if apply_basic_tokenization:
        tokenize_fn = tokenizer.basic_tokenizer.tokenize
    tokens = []
    for token in text.split(" "):
        if _SPECIAL_TOKENS_RE.match(token):
            if token in tokenizer.vocab:
                tokens.append(token)
            else:
                tokens.append(tokenizer.wordpiece_tokenizer.unk_token)
        else:
            tokens.extend(tokenize_fn(token))
    return tokens

def convert_single_example(example, tokenizer, dataset_type):
    """Converts a single NqExample into a list of InputFeatures."""
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    features = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenize(tokenizer, token)
        tok_to_orig_index.extend([i] * len(sub_tokens))
        all_doc_tokens.extend(sub_tokens)

    # `tok_to_orig_index` maps wordpiece indices to indices of whitespace
    # tokenized word tokens in the contexts. The word tokens might themselves
    # correspond to word tokens in a larger document, with the mapping given
    # by `doc_tokens_map`.
    if example.doc_tokens_map:
        tok_to_orig_index = [example.doc_tokens_map[index] for index in tok_to_orig_index]

    # QUERY
    query_tokens = []
    query_tokens.append("[Q]")
    query_tokens.extend(tokenize(tokenizer, example.questions[-1]))
    if len(query_tokens) > nq_config.max_question_length:
        query_tokens = query_tokens[-nq_config.max_question_length:]

    # ANSWER
    tok_start_position = 0
    tok_end_position = 0
    if dataset_type == 'train':
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = nq_config.max_context_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        length = min(length, max_tokens_for_doc)
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, nq_config.window_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        tokens.extend(query_tokens)
        segment_ids.extend([0] * len(query_tokens))
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = check_is_max_context(doc_spans, doc_span_index,
                                                                                        split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        assert len(tokens) == len(segment_ids)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (nq_config.max_context_length - len(input_ids))
        input_ids.extend(padding)
        input_mask.extend(padding)
        segment_ids.extend(padding)

        assert len(input_ids) == nq_config.max_context_length
        assert len(input_mask) == nq_config.max_context_length
        assert len(segment_ids) == nq_config.max_context_length

        start_position = None
        end_position = None
        answer_type = None
        answer_text = ""
        if dataset_type == 'train':
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            contains_an_annotation = (tok_start_position >= doc_start and tok_end_position <= doc_end)
            if ((not contains_an_annotation) or
                    example.answer.type == AnswerType.UNKNOWN):
                # If an example has unknown answer type or does not contain the answer
                # span, then we only include it with probability --include_unknowns.
                # When we include an example with unknown answer type, we set the first
                # token of the passage to be the annotated short span.
                if (nq_config.include_unknowns < 0 or
                        random.random() > nq_config.include_unknowns):
                    continue
                start_position = 0
                end_position = 0
                answer_type = AnswerType.UNKNOWN
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
                answer_type = example.answer.type

            answer_text = " ".join(tokens[start_position:(end_position + 1)])

        feature = InputFeatures(unique_id=-1,
                                example_index=-1,
                                doc_span_index=doc_span_index,
                                tokens=tokens,
                                token_to_orig_map=token_to_orig_map,
                                token_is_max_context=token_is_max_context,
                                input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                start_position=start_position,
                                end_position=end_position,
                                answer_text=answer_text,
                                answer_type=answer_type)

        features.append(feature)

    return features

def convert_examples_to_features_and_save(examples, tokenizer, dataset_type, features_file, n_examples):
    """Converts a list of NqExamples into InputFeatures."""
    num_spans_to_ids = collections.defaultdict(list)
    #import pdb; pdb.set_trace()

    with h5py.File(features_file, "w") as outfile:
        # Create h5py file with maximum possible size
        dset_size = n_examples * nq_config.max_windows
        unique_ids = outfile.create_dataset("unique_ids", (dset_size,), maxshape=(dset_size,), dtype='int64') # Necessary to prevent overflow
        input_ids = outfile.create_dataset("input_ids", (dset_size, nq_config.max_context_length,), maxshape=(dset_size, nq_config.max_context_length,), dtype='i')
        input_mask = outfile.create_dataset("input_mask", (dset_size, nq_config.max_context_length,), maxshape=(dset_size, nq_config.max_context_length,), dtype='i')
        segment_ids = outfile.create_dataset("segment_ids", (dset_size, nq_config.max_context_length,), maxshape=(dset_size, nq_config.max_context_length,), dtype='i')
        if dataset_type == 'train':
            start_positions = outfile.create_dataset("start_positions", (dset_size,), maxshape=(dset_size,), dtype='i')
            end_positions = outfile.create_dataset("end_positions", (dset_size,), maxshape=(dset_size,), dtype='i')
            answer_types = outfile.create_dataset("answer_types", (dset_size,), maxshape=(dset_size,), dtype='i')
        else:
            token_map = outfile.create_dataset("token_map", (dset_size, nq_config.max_context_length,), maxshape=(dset_size, nq_config.max_context_length,), dtype='i')

        with tqdm(total=n_examples) as pbar:
            i = 0
            for example in examples:
                example_index = example.example_id
                features = convert_single_example(example, tokenizer, dataset_type)
                num_spans_to_ids[len(features)].append(example.qas_id)

                for feature in features:
                    feature.example_index = example_index
                    feature.unique_id = feature.example_index + feature.doc_span_index

                    unique_ids[i] = np.array(feature.unique_id)
                    input_ids[i,:] = np.array(feature.input_ids)
                    input_mask[i,:] = np.array(feature.input_mask)
                    segment_ids[i,:] = np.array(feature.segment_ids)
                    if dataset_type == 'train':
                        start_positions[i] = np.array(feature.start_position)
                        end_positions[i] = np.array(feature.end_position)
                        answer_types[i] = np.array(feature.answer_type)
                    else:
                        token_map_list = [-1] * len(input_ids[i,:])
                        for k, v in feature.token_to_orig_map.items():
                            token_map_list[k] = v
                        token_map[i,:] = np.array(token_map_list)
                    
                    i += 1
                pbar.update(1)

        # Resize back to actual size
        unique_ids.resize(i, axis=0)
        input_ids.resize(i, axis=0)
        input_mask.resize(i, axis=0)
        segment_ids.resize(i, axis=0)
        if dataset_type == 'train':
            start_positions.resize(i, axis=0)
            end_positions.resize(i, axis=0)
            answer_types.resize(i, axis=0)
        else:
            token_map.resize(i, axis=0)

    return i, num_spans_to_ids

class NQFeatures(Dataset):
    """Input features for the Natural Questions as a PyTorch dataset."""
    def __init__(self, features_file, dataset_type):
        self.features_file = features_file
        self.dataset_type = dataset_type

        self.instance_types = dict()
        with h5py.File(self.features_file, "r") as f:
            ids = f['unique_ids']
            self.length = len(ids)
        #self.length = n_features

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.features_file, "r") as f:
            unique_id = torch.tensor(f['unique_ids'][idx], dtype=torch.long) 
            input_ids = torch.tensor(f['input_ids'][idx], dtype=torch.long)
            attention_mask = torch.tensor(f['input_mask'][idx], dtype=torch.long)
            token_type_ids = torch.tensor(f['segment_ids'][idx], dtype=torch.long)
            cls_index = torch.tensor(0, dtype=torch.long)
            if self.dataset_type != 'train':
                example_index = torch.tensor(idx, dtype=torch.long)
                token_map = torch.tensor(f['token_map'][idx], dtype=torch.long)
                sample = {'unique_id':unique_id,
                          'input_ids':input_ids,
                          'attention_mask':attention_mask,
                          'token_type_ids':token_type_ids,
                          'example_index':example_index,
                          'cls_index':cls_index,
                          'token_map':token_map}
            else:
                start_positions = torch.tensor(f['start_positions'][idx], dtype=torch.long)
                end_positions = torch.tensor(f['end_positions'][idx], dtype=torch.long)
                instance_types = torch.tensor(f['answer_types'][idx], dtype=torch.long)

                sample = {'unique_id':unique_id,
                          'input_ids':input_ids,
                          'attention_mask':attention_mask,
                          'token_type_ids':token_type_ids,
                          'start_positions':start_positions,
                          'end_positions':end_positions,
                          'instance_types':instance_types,
                          'cls_index':cls_index}
            return sample

def load_or_precompute_nq_features(args, tokenizer, dataset_type='test'):
    '''Precompute features and save them to hdf5, or load precomputed features from file.'''
    input_file = args.dev_file if dataset_type == 'dev' else args.train_file if dataset_type == 'train' else args.test_file
    input_file = Path(input_file)
    features_file = args.dev_features if dataset_type == 'dev' else args.train_features if dataset_type == 'train' else args.test_features
    features_file = Path(features_file)

    if not features_file.is_file() and not args.overwrite_cached_input_features:
        logger.info("Computing features and saving to input features file %s", features_file)
        if args.local_rank in [-1, 0]:
            n_eval_examples, eval_examples = read_nq_examples(input_file=input_file, dataset_type=dataset_type)
            n_features, num_spans_to_ids = convert_examples_to_features_and_save(examples=eval_examples,
                                                                                 tokenizer=tokenizer,
                                                                                 dataset_type=dataset_type,
                                                                                 features_file=features_file,
                                                                                 n_examples=n_eval_examples)
            logger.info("Saved input features to dir %s", features_file)
        return NQFeatures(features_file, dataset_type=dataset_type)
    else:
        logger.info("Loading %s features from file  %s", dataset_type, features_file)
        return NQFeatures(features_file, dataset_type=dataset_type)

