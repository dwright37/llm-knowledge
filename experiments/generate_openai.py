from llm_knowledge.model import get_model
from llm_knowledge.generate import bulk_generate
import argparse
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
    parser.add_argument("--output_dir", type=str, help="Name of the directory to save output",
                        required=True)
    parser.add_argument("--spec_file", type=str, help="A json file with a list of models and topics to generate from", required=True)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)

    args = parser.parse_args()

    enforce_reproducibility(args.seed)
    seed = args.seed
    spec_file = args.spec_file
    output_dir = Path(f"{args.output_dir}/generate")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(spec_file) as f:
        generation_spec = json.load(f)

    if isinstance(generation_spec['topics'], str):
        with open(generation_spec['topics']) as f:
            issues = [l.strip() for l in f.readlines()]
    else:
        issues = generation_spec['topics']

    if isinstance(generation_spec['prompt_templates'], str):
        with open(generation_spec['prompt_templates']) as f:
            prompts = json.load(f)['templates']
    else:
        prompts = generation_spec['prompt_templates']

    rag_data = None
    if 'rag_file' in generation_spec:
        with open(generation_spec['rag_file']) as f:
            rag_data = json.load(f)
    model_ids = generation_spec['models']
    random_samples = generation_spec['random_samples'] if 'random_samples' in generation_spec else 1
    temperatures = generation_spec['temperatures'] if 'temperatures' in generation_spec else [1.0]

    download_dir = None
    if 'HF_HOME' in os.environ:
        download_dir = os.environ['HF_HOME']

    model = None
    logfile = f"{output_dir}/models_completed.txt"
    if not os.path.exists(logfile):
        open(logfile, 'w')

    with open(logfile, 'r') as f:
        finished_models = set(f.read().splitlines())

    for model_id in model_ids:
        if model_id in finished_models:
            continue
        model = get_model(
            model_id,
            seed=seed,
            backend='openai',
            batched=True,
            openai_credentials={
                "api_key": "Your key",
                "project": "Your project ID",
                "organization": "Your organization ID",
            }
        )

        # We can run everything at once since we're batching
        complete_dframe = bulk_generate(
            issues,
            model,
            user_prompt_variations=prompts,
            topic_ids=list(range(len(issues))),
            n_samples=random_samples,
            temperatures=temperatures,
            rag_context=rag_data,
            max_new_tokens=2100
        )


        # Generate a lot of responses
        for k,issue in enumerate(issues):
            outfilename = f"{output_dir}/{model_id.replace('/', '_')}/{issue.replace('/', '_')}.pqt"
            if not os.path.exists(f"{output_dir}/{model_id.replace('/', '_')}"):
                os.makedirs(f"{output_dir}/{model_id.replace('/', '_')}")
            if os.path.exists(outfilename):
                continue

            responses_dframe = complete_dframe[complete_dframe['topic_id'] == k].reset_index(drop=True)
            # Add the model name
            responses_dframe['model_id'] = [model_id] * len(responses_dframe)

            out_issues = [issues[id_] for id_ in responses_dframe['topic_id']]
            responses_dframe['topic'] = out_issues
            responses_dframe = responses_dframe.reset_index(drop=True)
            responses_dframe.to_parquet(outfilename, index=None)
        finished_models.add(model_id)
        with open(logfile, 'a') as f:
            f.write('\n' + model_id)