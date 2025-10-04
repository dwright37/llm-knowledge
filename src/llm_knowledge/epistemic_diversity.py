import random
import pandas as pd
from typing import List, Dict, Tuple, AnyStr
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.cluster import DBSCAN
import numpy as np
from collections import Counter
from torch import cuda
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from os import path
from collections import defaultdict
resources_dir = path.dirname(__file__)
import nltk
import time
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
from tqdm import tqdm
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyPairDataset
import spacy
nlp = spacy.load("en_core_web_sm", disable=['ner'])
import gc
import torch

from .model import QueryModel
import re

MIN_CLUSTER_SIZE = 10
TEXT_COLUMN = "response"


MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DEVICE = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
SYSTEM_PROMPT_PATH = f"{resources_dir}/prompts/epistemic_diversity/system_prompt.txt"
USER_PROMPT_PATH = f"{resources_dir}/prompts/epistemic_diversity/user_prompt.txt"
ASSISTANT_PROMPT_PATH = f"{resources_dir}/prompts/epistemic_diversity/assistant_prompt.txt"
STANCE_PROMPT_PATH = f"{resources_dir}/prompts/stance_prompt.txt"
ARGUMENT_SUPPORT_PROMPT_PATH = f"{resources_dir}/prompts/argument_support_prompt.txt"
FACTOID_EXTRACTION_PROMPT_PATH = f"{resources_dir}/prompts/epistemic_diversity/factoid_extraction_prompt.txt"
with open(FACTOID_EXTRACTION_PROMPT_PATH) as f:
    FACTOID_EXTRACTION_PROMPT = f.read().strip()


def embed_sentences(
    sentences: List[str],
    model: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Function to embed the sentences using the  sentence-transformer given model

    :param sentences: A list of strings to be embedded
    :param model: A string with the ID of the sentence-transformers model to use
    :return: A matrix of shape (n_sentence, embedding_dim) containing the embeddings
    """
    embedder = SentenceTransformer(
        model, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    embeddings = embedder.encode(sentences, show_progress_bar=True)
    return embeddings


def cluster_sentences(
    embeddings: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 10
) -> np.ndarray:
    """
    Function to cluster the embeddings using DBSCAN

    :param embeddings: Numpy array with embeddings to cluster
    :param eps: eps parameter for DBSCAN
    :param min_samples: min_samples param for DBSCAN
    :return: Cluster membership of each point in the embedding list
    """
    db = DBSCAN(metric="cosine", eps=eps, min_samples=min_samples)
    clusters = db.fit_predict(embeddings)
    print(f"Number of clusters: {len(set(clusters))}")
    return clusters


def filter_dframe(
        data: pd.DataFrame,
        group_keys: List[str] = None,
        group_values: List[str] = None
):
    curr_data = data
    if len(group_keys) > 0:
        assert len(group_values) == len(group_keys), "group_value must be provided if group_key is provided"
        for group_key, group_value in zip(group_keys, group_values):
            # go through the clusters and calculate their weight
            curr_data = curr_data[curr_data[group_key] == group_value]
    return curr_data


def calculate_probabilities(
        data: pd.DataFrame,
        sampled_data: pd.DataFrame = None,
        group_keys: List[str] = None,
        group_values: List[str] = None,
        return_type: str = 'dict'
) -> pd.DataFrame:
    """
    Given a clustered dataframe, calculate the probability as in the semantic entropy paper
    :param data: A dataframe.
    :param group_key: A column name to use for grouping clusters
    :return: A dict mapping trope name to its probability
    """

    all_clusters = set(data['cluster'].to_list())
    if sampled_data is not None:
        curr_data = sampled_data
    else:
        curr_data = filter_dframe(data, group_keys=group_keys, group_values=group_values)

    counts = Counter(curr_data['cluster'])
    scores = {c: counts[c] / len(curr_data) if len(curr_data) > 0 else 0.0 for c in all_clusters}
    # for cluster in all_clusters:
    #     scores[cluster] = len(curr_data[curr_data['cluster'] == cluster]) / len(curr_data)
    if len(curr_data) > 0:
        assert np.round(sum(scores.values()), 5) == 1.

    if return_type == 'dict':
        return scores, len(curr_data)
    else:
        return np.array([scores[t] for t in sorted(scores.keys())]), len(curr_data)


def calculate_entropies_progressive(
        data: pd.DataFrame,
        group_keys: List[str] = None,
        group_values: List[str] = None,
        return_type: str = 'dict'
) -> List:
    """
    Given a clustered dataframe, calculate all of the entropies based on adding new prompts
    :param data: A dataframe.
    :param group_key: A column name to use for grouping clusters
    :return: A dict mapping trope name to its probability
    """

    all_clusters = set(data['cluster'].to_list())
    all_data = data
    if len(group_keys) > 0:
        assert len(group_values) == len(group_keys), "group_value must be provided if group_key is provided"
        for group_key, group_value in zip(group_keys, group_values):
            # go through the clusters and calculate their weight
            all_data = all_data[all_data[group_key] == group_value]

    scores = defaultdict(int)
    total = 0
    entropies = []
    for k in tqdm(sorted(list(set(all_data['original_index'].to_list())))):
        curr_data = all_data[all_data['original_index'] == k]
        curr_counts = Counter(curr_data['cluster'])
        for cluster in all_clusters:
            scores[cluster] = ((scores[cluster] * total) + curr_counts[cluster]) / (total + len(curr_data))
        total += len(curr_data)
        assert np.round(sum(scores.values()), 5) == 1.
        entropies.append(calculate_entropy(scores))

    return entropies


def calculate_entropy(
        probabilities
) -> float:
    if isinstance(probabilities, dict):
        probabilities = list(probabilities.values())
    return entropy(probabilities) if sum(probabilities) != 0 else 0.


def calculate_evenness(
        probabilities
):
    # Get the actual entropy
    ent = calculate_entropy(probabilities)
    # Get the max entropy
    maxent = np.log(len(probabilities))

    return ent / maxent


def calculate_evenness_statement(
        data: pd.DataFrame,
        group_keys: List[str] = None,
        group_values: List[str] = None,
        return_type: str = 'dict'
):
    probabilities, n_statements = calculate_probabilities(data, group_keys, group_values, return_type)
    # Get the actual entropy
    ent = calculate_entropy(probabilities)
    # Get the max entropy
    maxent = np.log(len(probabilities))

    return ent / maxent


def estimate_coverage(
        data: pd.DataFrame,
        group_keys: List[str] = None,
        group_values: List[str] = None
):
    """
    Reference: https://esajournals.onlinelibrary.wiley.com/doi/10.1890/11-1952.1
    Originally from Chen and Shao, 2010
    :param data:
    :param group_keys:
    :param group_values:
    :param return_type:
    :return:
    """
    if group_keys != None:
        dframe = filter_dframe(data, group_keys=group_keys, group_values=group_values)
    else:
        dframe = data
    # Count singletons
    cluster_counts = Counter(dframe['cluster'])
    f1 = len([c for c in cluster_counts if cluster_counts[c] == 1])
    # Count duplicates
    f2 = len([c for c in cluster_counts if cluster_counts[c] == 2])
    # Total statements
    n = len(dframe)

    term1 = (n-1)*f1
    return 1 - (f1 / n)*(term1 / (term1 + 2*f2))


def resample_to_coverage_level(
        data: pd.DataFrame,
        group_keys: List[str] = None,
        group_values: List[str] = None,
        return_type: str = 'dict',
        coverage_level: float = 0.5
):
    data = filter_dframe(data, group_keys=group_keys, group_values=group_values)
    current_coverage = estimate_coverage(data)
    out_data = data
    n = len(out_data)
    idxs = list(range(n))
    if current_coverage > coverage_level:
        # Binary search
        cov = 0
        rng = list(range(n))
        while len(rng) > 2 and abs(coverage_level - cov) > 0.001:
            # get the halfway point
            midx = int(len(rng) / 2)
            m = rng[midx]
            # Take a sample, calculate coverage
            out_data = data.iloc[list(random.sample(idxs, m))]
            cov = estimate_coverage(out_data)
            if cov > coverage_level:
                rng = rng[:midx]
            else:
                rng = rng[midx:]
    return out_data


def calculate_diversity(
        data: pd.DataFrame,
        sampled_data: pd.DataFrame = None,
        group_keys: List[str] = None,
        group_values: List[str] = None,
        return_type: str = 'dict'
):
    probabilities, n_statements = calculate_probabilities(data, sampled_data, group_keys, group_values, return_type)
    # Get the actual entropy
    ent = calculate_entropy(probabilities)

    hillshannon = np.exp(ent) if ent > 0 else 0.
    return ent,hillshannon,probabilities


def calculate_divergence(
        p1,
        p2
):
    assert len(p1) == len(p2)
    return jensenshannon(p1, p2)


def chunk_document(document, tokenizer, N=3, max_tokens=2000):
    # spacy has a max input length of 100000 chars
    for m in range(0, len(document), 100000):
        text = document[m:m+100000]
        sentences = list(nlp(text).sents)
        for k in range(0, len(sentences), N):
            original_input = ' '.join([s.text for s in sentences[k:k+N]])
            if tokenizer:
                tokenized = tokenizer.encode(original_input, add_special_tokens=False)
                for j in range(0, len(tokenized), max_tokens):
                    yield tokenizer.decode(tokenized[j:j+max_tokens])
            else:
                yield original_input


def create_factoid_prompt(
        topic: str,
        response: str
) -> Dict:
    """
    Get model input for factoid extraction
    :param topic: The topic for the prompt
    :param response: The text to extract factoids from
    :return:
    """
    return [{
        "role": "user", "content": FACTOID_EXTRACTION_PROMPT.replace("{issue}", topic).replace("{content}", response)
    }]


def cluster_entailment_multiple_with_checkpointing(
        entailment_pipeline,
        original_dframe: pd.DataFrame,
        outfile_name: AnyStr,
        checkpoint_steps=0,
        N=50,
        sim_block_size=1000
) -> np.ndarray:
    """
    Function to cluster sentences using Natural Language Inference (NLI) based on mutual entailment

    :param entailment_pipeline: The NLI pipeline used to determine entailment between sentences
    :param embeddings: Numpy array with embeddings used to find similar sentences
    :param sentences: List of sentences to cluster
    :param N: Maximum number of hypotheses to consider for each sentence
    :return: Cluster membership of each sentence in the input list
    """
    sentences = original_dframe['factoid'].to_list()
    embeddings = embed_sentences(sentences)
    sent_to_attempt = defaultdict(set)

    if 'cluster' not in original_dframe.columns:
        original_dframe['cluster'] = -1

    original_dframe.loc[0, 'cluster'] = 0
    cluster_max = 1
    sentence_to_cluster = {}
    current_counts = 0
    for step in range(2):
        cluster_counts = Counter(original_dframe['cluster'].to_list())
        cluster_len = len(cluster_counts.keys())
        if step > 1 and current_counts - cluster_len <= 10:
            break
        current_counts = cluster_len
        idx_to_samples = {}
        # set up the inputs
        inputs = {
            'text': [],
            'text_pair': []
        }
        counter = 0
        for j, sent in enumerate(tqdm(sentences)):
            if j % sim_block_size == 0:
                # Get the cosine similarities (we do it in blocks in order to improve efficiency)
                similarity_block = cos_sim(embeddings[j:j + 1000], embeddings)
            # Only do singletons after the first round
            if cluster_counts[original_dframe.loc[j, 'cluster']] > 1 and step > 0:
                idx_to_samples[j] = None
            else:
                skip = False
                if step == 0:

                    # Skip sentences that are already clustered
                    if original_dframe.loc[j, 'cluster'] != -1:
                        sentence_to_cluster[sent] = original_dframe.loc[j, 'cluster']
                        skip = True
                        idx_to_samples[j] = None
                    sims = similarity_block[j % sim_block_size][:j + 1]
                else:
                    sims = similarity_block[j % sim_block_size]

                order = np.argsort(sims.numpy())[::-1]
                sort_order = order[order != j]
                hypo_idxs = []
                att = 0
                while len(hypo_idxs) < N and att < len(sort_order):
                    if sort_order[att] not in sent_to_attempt[j]:
                        hypo_idxs.append(sort_order[att])
                    att += 1
                sent_to_attempt[j].update(hypo_idxs)
                if not skip:
                    hypotheses = [sentences[k] for k in hypo_idxs]
                    idx_to_samples[j] = []
                    for hypo in hypotheses:
                        inputs['text'].append(sent)
                        inputs['text_pair'].append(hypo)
                        inputs['text'].append(hypo)
                        inputs['text_pair'].append(sent)
                        idx_to_samples[j].append(counter)
                        idx_to_samples[j].append(counter + 1)
                        counter += 2
            if j % checkpoint_steps == 0 or j == len(sentences) - 1:
                dset = Dataset.from_dict(inputs)
                all_labels = [l for l in tqdm(entailment_pipeline(KeyPairDataset(dset, 'text', 'text_pair'), batch_size=32), total=len(inputs['text']), desc="Performing NLI")]
                for sent_idx in tqdm(idx_to_samples):
                    if sent_idx % sim_block_size == 0:
                        # Get the cosine similarities (we do it in blocks in order to improve efficiency)
                        print(f"N clusters: {len(Counter(original_dframe['cluster'].to_list()).keys())}")
                    if idx_to_samples[sent_idx] != None:
                        found = []
                        entailment_counts = defaultdict(int)
                        labels = [all_labels[k] for k in idx_to_samples[sent_idx]]
                        for idx in range(0, len(labels), 2):
                            if labels[idx]['label'] == 'ENTAILMENT':
                                if labels[idx + 1]['label'] == 'ENTAILMENT':
                                    found.append((sentence_to_cluster[inputs['text_pair'][idx_to_samples[sent_idx][idx]]], np.log(labels[idx]['score']) + np.log(labels[idx + 1]['score'])))
                                else:
                                    # Unidirectional entailment, add to counter
                                    entailment_counts[sentence_to_cluster[inputs['text_pair'][idx_to_samples[sent_idx][idx]]]] += 1
                        if len(found) == 0:
                            if step == 0:
                                cluster = cluster_max
                                cluster_max += 1
                            else:
                                cluster = original_dframe.loc[sent_idx, 'cluster']
                        else:
                            cluster_idx = np.argmax([s for c,s in found])

                            cluster = found[cluster_idx][0]

                            del sent_to_attempt[sent_idx]
                    else:
                        cluster = original_dframe.loc[sent_idx, 'cluster']

                    original_dframe.loc[sent_idx, 'cluster'] = cluster
                    sentence_to_cluster[sentences[sent_idx]] = cluster
                # Checkpoint
                original_dframe.to_parquet(outfile_name, index=False)
                counter = 0
                idx_to_samples = {}
                # set up the inputs
                inputs = {
                    'text': [],
                    'text_pair': []
                }
                gc.collect()

    return original_dframe


def break_up_clusters(
        entailment_pipeline,
        original_dframe: pd.DataFrame,
        outfile_name: AnyStr,
        checkpoint_steps=0,
        N=50,
        sim_block_size=1000
) -> np.ndarray:
    """
    Function to cluster sentences using Natural Language Inference (NLI) based on mutual entailment

    :param entailment_pipeline: The NLI pipeline used to determine entailment between sentences
    :param embeddings: Numpy array with embeddings used to find similar sentences
    :param sentences: List of sentences to cluster
    :param N: Maximum number of hypotheses to consider for each sentence
    :return: Cluster membership of each sentence in the input list
    """
    print("break_up_clusters")
    # Iterate through all clusters which have more than 1000 members, break these into smaller clusters
    sentences = np.array(original_dframe['factoid'].to_list())
    embeddings = embed_sentences(sentences)
    clusters = Counter(original_dframe['cluster'].to_list())
    cluster_ptr = max(clusters.keys())
    eps_array = np.logspace(0.04, 0.1, 9901)
    # Iterate through clusters
    for cluster in tqdm(sorted(clusters, key=lambda x: clusters[x], reverse=True)):
        # Leave out everything below 100
        if clusters[cluster] < 100:
            break
        # Get all of the sentences, indices, and embeddings for this cluster
        cluster_dframe = original_dframe[original_dframe['cluster'] == cluster]
        indices = cluster_dframe.index.to_list()
        embeddings_curr = embeddings[indices]
        # Get what eps should be; max 0.1, min 0.04
        eps_idx = 9900 - (min(10000, clusters[cluster]) - 100)
        eps = np.log10(eps_array[eps_idx])
        db = DBSCAN(metric="cosine", eps=eps, min_samples=20)
        new_clusters = db.fit_predict(embeddings_curr)
        # Get a mapping from new cluster to old cluster
        cluster_counts = Counter(new_clusters)
        if len(cluster_counts) > 1:
            max_cluster = max([(c,v) for c,v in cluster_counts.items() if c != -1], key=lambda x: x[1])[0]
        else:
            max_cluster = list(cluster_counts.keys())[0] if -1 not in cluster_counts else 0
        cluster_map = {-1: -1}
        for c in cluster_counts:
            if c == max_cluster:
                cluster_map[c] = cluster
            elif c != -1:
                cluster_map[c] = cluster_ptr + 1
                cluster_ptr += 1
        # Now reassign all clusters
        for c,idx in zip(new_clusters, indices):
            original_dframe.loc[idx, 'cluster'] = cluster_map[c]
    # Now go and re-run NLI on the singletons again
    sentence_to_cluster = {row['factoid']: row['cluster'] for idx, row in original_dframe.iterrows()}
    sent_to_attempt = defaultdict(set)
    cluster_max = max(Counter(original_dframe['cluster'].to_list()).keys()) + 1
    current_counts = 0
    del db
    del cluster_dframe
    gc.collect()
    for step in range(2):
        cluster_counts = Counter(original_dframe['cluster'].to_list())
        cluster_len = len(cluster_counts.keys())
        if step > 1 and current_counts - cluster_len <= 10:
            break
        current_counts = cluster_len
        print(len(cluster_counts.keys()))
        idx_to_samples = {}
        # set up the inputs
        inputs = {
            'text': [],
            'text_pair': []
        }
        counter = 0
        sent_mask = original_dframe['cluster'].to_numpy() == -1
        for j, sent in enumerate(tqdm(sentences)):
            if j % sim_block_size == 0:
                # Get the cosine similarities (we do it in blocks in order to improve efficiency)
                similarity_block = cos_sim(embeddings[j:j + 1000], embeddings).numpy()
            # Only do singletons after the first round
            if cluster_counts[original_dframe.loc[j, 'cluster']] > 1 and step > 0:
                idx_to_samples[j] = None
            else:
                skip = False
                if step == 0:

                    if original_dframe.loc[j, 'cluster'] != -1:
                        sentence_to_cluster[sent] = original_dframe.loc[j, 'cluster']
                        skip = True
                        idx_to_samples[j] = None
                    # Use all of them, we'll skip the ones that aren't clustered
                    sims = similarity_block[j % sim_block_size]  # cos_sim(embeddings[j], embeddings[:j+1])
                else:
                    sims = similarity_block[j % sim_block_size]  # cos_sim(embeddings[j], embeddings)

                sims[sent_mask] = -10000
                sims[j] = -10000
                for att in sent_to_attempt[j]:
                    sims[att] = -10000
                hypo_idxs = list(np.argpartition(sims, -N)[-N:])

                sent_to_attempt[j].update(hypo_idxs)
                if not skip:
                    hypotheses = [sentences[k] for k in hypo_idxs]
                    idx_to_samples[j] = []
                    for hypo in hypotheses:
                        inputs['text'].append(sent)
                        inputs['text_pair'].append(hypo)
                        inputs['text'].append(hypo)
                        inputs['text_pair'].append(sent)
                        idx_to_samples[j].append(counter)
                        idx_to_samples[j].append(counter + 1)
                        counter += 2
            if j % checkpoint_steps == 0 or j == len(sentences) - 1:
                dset = Dataset.from_dict(inputs)
                all_labels = [l for l in
                              tqdm(entailment_pipeline(KeyPairDataset(dset, 'text', 'text_pair'), batch_size=32),
                                   total=len(inputs['text']), desc="Performing NLI")]
                for sent_idx in tqdm(idx_to_samples):

                    if idx_to_samples[sent_idx] != None:
                        found = []
                        entailment_counts = defaultdict(int)
                        labels = [all_labels[k] for k in idx_to_samples[sent_idx]]
                        for idx in range(0, len(labels), 2):
                            if labels[idx]['label'] == 'ENTAILMENT':
                                if labels[idx + 1]['label'] == 'ENTAILMENT':
                                    found.append(
                                        (sentence_to_cluster[inputs['text_pair'][idx_to_samples[sent_idx][idx]]],
                                         np.log(labels[idx]['score']) + np.log(labels[idx + 1]['score'])))
                                else:
                                    # Unidirectional entailment, add to counter
                                    entailment_counts[
                                        sentence_to_cluster[inputs['text_pair'][idx_to_samples[sent_idx][idx]]]] += 1
                        if len(found) == 0:

                            if step == 0:
                                cluster = cluster_max
                                cluster_max += 1
                            else:
                                cluster = original_dframe.loc[sent_idx, 'cluster']
                        else:
                            cluster_idx = np.argmax([s for c, s in found])

                            cluster = found[cluster_idx][0]
                            del sent_to_attempt[sent_idx]
                    else:
                        cluster = original_dframe.loc[sent_idx, 'cluster']

                    original_dframe.loc[sent_idx, 'cluster'] = cluster
                    sentence_to_cluster[sentences[sent_idx]] = cluster
                # Checkpoint
                original_dframe.to_parquet(outfile_name, index=False)
                counter = 0
                idx_to_samples = {}
                # set up the inputs
                inputs = {
                    'text': [],
                    'text_pair': []
                }
                sent_mask = original_dframe['cluster'].to_numpy() == -1
                gc.collect()
    return original_dframe


def extract_claims_bulk(
        generation_dframe: pd.DataFrame,
        extraction_model: QueryModel,
        group_key: str = "stimulus_id",
        prompt: str = FACTOID_EXTRACTION_PROMPT,
):
    all_dframes = []

    # Iterate through groups
    for name, group in generation_dframe.groupby(group_key):
        # Get the tokens
        responses = group['text'].tolist()
        topics = group['topic'].tolist()
        msgs = []
        chunk_topics = []
        chunks = []
        model_ids = []
        idxs = []
        for k,resp in enumerate(tqdm(responses)):
            for chunk in chunk_document(resp, extraction_model.tokenizer, max_tokens=300):
                msgs.append([{
                    "role": "user", "content": prompt.replace("{issue}", topics[k]).replace("{content}", chunk)
                }])

                chunks.append(chunk)
                chunk_topics.append(name)
                model_ids.append(group.iloc[k]['model_id'])
                idxs.append(group.index[k])
        outputs = extraction_model.generate(
            msgs,
            max_new_tokens=700,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
            n_samples=1
        )
        # See if this helps with memory issue
        time.sleep(5)
        gc.collect()
        torch.cuda.empty_cache()

        # Get all of the factoids from each output
        factoids = []
        out_topics = []
        out_chunks = []
        out_idxs = []
        out_model_ids = []
        for j,out in enumerate(outputs['text']):
            factoids_curr = [s.strip() for s in re.split(r'\n+', out) if len(s.strip()) > 0]
            out_topics.extend([chunk_topics[j]]*len(factoids_curr))
            out_chunks.extend([chunks[j]] * len(factoids_curr))
            out_idxs.extend([idxs[j]] * len(factoids_curr))
            out_model_ids.extend([model_ids[j]] * len(factoids_curr))
            factoids.extend(factoids_curr)

        out_dframe = pd.DataFrame()
        out_dframe['group'] = [name] * len(factoids)
        out_dframe['chunk'] = out_chunks
        out_dframe['topic'] = out_topics
        out_dframe['factoid'] = factoids
        out_dframe['model_id'] = out_model_ids
        out_dframe['original_index'] = out_idxs


        all_dframes.append(out_dframe)
    all_data_df = pd.concat(all_dframes, ignore_index=True)
    return all_data_df
