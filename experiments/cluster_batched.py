from llm_knowledge.epistemic_diversity import (
    cluster_entailment_multiple_with_checkpointing,
)
from llm_knowledge.model import get_model
from transformers import pipeline
import argparse
import pandas as pd
import torch
import numpy as np
import random
import json
import os
from pathlib import Path
import glob
import gc
from tqdm import tqdm


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


def load_input_data(input_dirs, topic):
    dframes = []
    # The final directory in the path gives us the setting (ift, rag, etc.)
    for dir in input_dirs:
        type = Path(dir).stem
        for fname in tqdm(list(glob.glob(f"{dir}/**/{topic.replace('/', '_')}.pqt", recursive=True)), desc="Loading files"):
            topic_dframe = pd.read_parquet(fname)
            assert 'model_id' in topic_dframe.columns, "Data frames must indicate the model ID!"
            assert 'factoid' in topic_dframe.columns, "Data frames must contain a factoid column!"
            assert 'topic' in topic_dframe.columns, "Data frames must specify the topic!"
            #topic_dframe = dframe[dframe['topic'] == topic].reset_index()
            # Filter short factoids
            topic_dframe = topic_dframe[topic_dframe['factoid'].map(len) > 15]
            topic_dframe['original_path'] = [fname] * len(topic_dframe)
            topic_dframe['setting'] = [type] * len(topic_dframe)
            dframes.append(topic_dframe)
    input_data = pd.concat(dframes)
    input_data = input_data.sample(frac=1).reset_index(drop=True)
    return input_data


if __name__ == '__main__':
    # Initialize the process group
    # dist.init_process_group(backend='nccl')

    # # Determine local rank and set device
    # local_rank = int(os.getenv('LOCAL_RANK', 0))
    # torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Name of the directory to save output and perform checkpointing",
                        required=True)
    parser.add_argument("--input_dirs", type=str, help="Name of the input directories to crawl parquet files from",
                        required=True, nargs='+')
    parser.add_argument("--append_dirs", type=str, help="Name of optional directories to append factoids to existing clusters",
                        default=[], nargs='+')
    parser.add_argument("--topic", type=str, help="The name of the topic to perform clustering for",
                        required=True)

    args = parser.parse_args()

    enforce_reproducibility(1000)
    input_dirs = args.input_dirs
    topic = args.topic.strip()
    topic_fname = topic.replace('/', '_')

    if Path(f"{args.output_dir}/{topic_fname}.done").exists():
        exit()

    output_file = Path(f"{args.output_dir}/{topic_fname}.pqt")
    if not os.path.exists(output_file.parent):
        os.makedirs(output_file.parent)

    # Open and combine all of the input files
    if not os.path.exists(output_file):
        input_data = load_input_data(input_dirs, topic)
    else:
        # Load from checkpoint
        input_data = pd.read_parquet(output_file)
        if len(args.append_dirs) > 0:
            append_data = load_input_data(args.append_dirs)
            append_data['cluster'] = [-1]*len(append_data)
            input_data = pd.concat([input_data, append_data])

    # Initialize the original pipeline
    pipe = pipeline("text-classification", model="microsoft/deberta-large-mnli")

    #group = group.sample(frac=0.05).reset_index(drop=True)

    # Original clustering method (commented out but kept for reference)
    out_dframe = cluster_entailment_multiple_with_checkpointing(
        pipe,
        input_data,
        outfile_name=output_file,
        checkpoint_steps=50000,
        N=6
    )

    out_dframe.to_parquet(output_file, index=False)

    # with open(f"{args.output_dir}/completed.txt", 'a') as f:
    #     f.write(f"\n{topic}\n")
    Path(f"{args.output_dir}/{topic_fname}.done").touch()