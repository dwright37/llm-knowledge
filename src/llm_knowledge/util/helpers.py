import json
import numpy as np
import random
import hashlib
import uuid
from copy import copy
from typing import List, Tuple
import spacy
nlp = spacy.load("en_core_web_sm")
# Has to come after spacy import to fix bug with vLLM
import torch
torch.cuda.current_device()


def create_uuid_from_string(val: str):
    if not isinstance(val, str):
        val = json.dumps(val)
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return uuid.UUID(hex=hex_string)


def enforce_reproducibility(seed=1000):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)


def get_token_offsets(tokens):
    offsets = []
    pos = 0
    for token in tokens:
        offsets.append((pos, pos+len(token)))
        pos += len(token)
    return offsets


def get_sentence_token_offsets(
        tokens,
        offsets = None
):
    if offsets == None:
        offsets = get_token_offsets(tokens)

    original_text = ''.join(tokens)
    k = 0
    sent_token_offests = []
    sentences = list(nlp(original_text).sents)
    for sent in sentences:
        start = copy(k)
        while k < len(offsets):
            if offsets[k][0] <= sent.end_char <= offsets[k][1]:
                k += 1
                break
            k += 1
        end = copy(k)
        sent_token_offests.append((start, end))
    assert len(sent_token_offests) == len(sentences)
    return sent_token_offests


def calculate_sentence_probabilities(
        logprobs: List[str],
        segment_offsets: List[Tuple]
):

    # Segment probabilities only account for the individual segment; sequence
    # probabilities consider the likelihood of the whole sequence up through the sentence
    logprobs = eval(logprobs)
    segment_probs = []
    segment_N = []
    sequence_probs = []
    sequence_N = []
    for offs in segment_offsets:
        segment_probs.append(np.mean([lp for lp in logprobs[offs[0]:offs[1]]]))
        segment_N.append(offs[1] - offs[0])
        sequence_probs.append(segment_probs[-1] if len(sequence_probs) == 0 else np.mean([lp for lp in logprobs[:offs[1]]]))
        sequence_N.append(offs[1])

    return segment_probs, segment_N, sequence_probs, sequence_N