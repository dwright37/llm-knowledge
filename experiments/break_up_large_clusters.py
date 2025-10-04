from llm_knowledge.epistemic_diversity import (
    break_up_clusters
)
from transformers import pipeline
import argparse
import pandas as pd
import torch
import numpy as np
import random
import os
from pathlib import Path
import glob
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
        for fname in tqdm(list(glob.glob(f"{dir}/**/{topic}.pqt", recursive=True)), desc="Loading files"):
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
    parser.add_argument("--output_file", type=str, help="Name of the output file to save to post-processed clusters",
                        required=True)
    parser.add_argument("--input_file", type=str, help="Name of the input file with the original clusters",
                        required=True)

    args = parser.parse_args()

    enforce_reproducibility(1000)
    input_file = args.input_file
    indonefile = input_file.replace(".pqt", ".done")

    # with open(f"{args.output_dir}/completed.txt") as f:
    #     completed = set([l.strip() for l in f if l.strip()])
    #     if topic in completed:
    #         exit()
    output_file = Path(args.output_file)
    if not os.path.exists(output_file.parent):
        os.makedirs(output_file.parent)

    donefile = args.output_file.replace('.pqt', '.done')
    if Path(donefile).exists() or not Path(indonefile).exists():
        exit()

    # Open and combine all of the input files
    input_data = pd.read_parquet(input_file)

    # Initialize the original pipeline
    pipe = pipeline("text-classification", model="microsoft/deberta-large-mnli")

    #group = group.sample(frac=0.05).reset_index(drop=True)

    # Original clustering method (commented out but kept for reference)
    #out_dframe = cluster_entailment_multiple_with_checkpointing_v5(pipe, input_data, outfile_name=output_file, checkpoint_steps=50000, N=10)
    out_dframe = break_up_clusters(
        pipe,
        input_data,
        outfile_name=output_file,
        checkpoint_steps=50000,
        N=6
    )
    out_dframe.to_parquet(output_file, index=False)

    # with open(f"{args.output_dir}/completed.txt", 'a') as f:
    #     f.write(f"\n{topic}\n")
    Path(donefile).touch()