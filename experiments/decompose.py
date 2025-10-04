from llm_knowledge.model import get_model
from llm_knowledge.epistemic_diversity import extract_claims_bulk, resources_dir
import argparse
import pandas as pd
import torch
import numpy as np
import random
import os
from pathlib import Path
import json


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


if __name__ == '__main__':
    # Initialize the process group
    # dist.init_process_group(backend='nccl')

    # # Determine local rank and set device
    # local_rank = int(os.getenv('LOCAL_RANK', 0))
    # torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Name of the input directory where all of the files are located",
                        required=True)
    parser.add_argument("--spec_file", type=str, help="A json file with a list of models and topics to generate from", required=True)

    parser.add_argument("--output_dir", type=str, help="Name of the file to save output",
                        required=True)
    parser.add_argument("--decomposition_model_id", type=str, help="The name of the model to use", default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--prompt_location", type=str, default=f"{resources_dir}/prompts/epistemic_diversity/factoid_extraction_prompt.txt")

    args = parser.parse_args()

    enforce_reproducibility(args.seed)
    seed = args.seed
    spec_file = args.spec_file
    output_dir = Path(f"{args.output_dir}/decompose")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(spec_file) as f:
        spec = json.load(f)

    if isinstance(spec['topics'], str):
        with open(spec['topics']) as f:
            issues = [l.strip() for l in f.readlines()]
    else:
        issues = spec['topics']

    model_ids = spec['models']
    decomp_model_id = args.decomposition_model_id

    download_dir = None
    if 'HF_HOME' in os.environ:
        download_dir = os.environ['HF_HOME']

    # Create a model
    model = get_model(
        decomp_model_id,
        seed=seed,
        backend='vllm',
        max_model_len = 2420, # Includes the input prompt length (up to 1120), max input length (up to 300), and max output length (1000)
        download_dir=download_dir,
        gpu_memory_utilization=0.9,
        max_num_seqs=128
    )

    with open(args.prompt_location) as f:
        FACTOID_EXTRACTION_PROMPT = f.read().strip()

    logfile = f"{output_dir}/models_completed.txt"
    if not os.path.exists(logfile):
        open(logfile, 'w')

    with open(logfile, 'r') as f:
        finished_models = set(f.read().splitlines())
    for model_id in model_ids:
        if model_id in finished_models:
            continue
        for issue in issues:
            outfilename = f"{output_dir}/{model_id.replace('/', '_')}/{issue.replace('/', '_')}.pqt"
            if not os.path.exists(f"{output_dir}/{model_id.replace('/', '_')}"):
                os.makedirs(f"{output_dir}/{model_id.replace('/', '_')}")
            if os.path.exists(outfilename):
                continue

            input_file = f"{args.input_dir}/{model_id.replace('/', '_')}/{issue.replace('/', '_')}.pqt"
            print(f"Processing {input_file}")
            input_path = Path(input_file)
            input_data = pd.read_parquet(input_file)
            # Decompose the responses
            responses_dframe = extract_claims_bulk(
                input_data,
                model,
                group_key='topic',
                prompt=FACTOID_EXTRACTION_PROMPT
            )
            responses_dframe = responses_dframe.reset_index(drop=True)
            responses_dframe.to_parquet(outfilename, index=None)
        with open(logfile, 'a') as f:
            f.write('\n' + model_id)