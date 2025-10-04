from llm_knowledge.epistemic_diversity import embed_sentences
from sentence_transformers.util import cos_sim
from tiktoken import get_encoding
import argparse
import torch
import numpy as np
import random
from pathlib import Path
import json
from collections import defaultdict
from tqdm import tqdm


# Empirically found by taking all cosine similarities and selecting the median value
MIN_COS_SIM = 0.34399465


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

def perform_rag(rag_file, issues, prompts):
    # Create a vector store of all of the retrieval text
    encoding = get_encoding("cl100k_base")
    with open(rag_file) as f:
        original_search = json.load(f)['queries']
    output_context = defaultdict(dict)
    cos_sims = defaultdict(list)
    tok_lens = []
    for issue in tqdm(issues, desc="Performing RAG"):
        if issue not in original_search:
            output_context[issue] = []
            continue

        for page in original_search[issue]:
            tok_lens.append(len(encoding.encode(page['text'])))
        chunks = ''.join([page['text'] for page in original_search[issue]]).split('\n')
        chunks = list(set([c for c in chunks if len(c) > 250]))
        chunk_embeddings = embed_sentences(chunks)
        prompt_text = [p.replace('{proposition}', issue) for p in prompts]
        prompt_embeddings = embed_sentences(prompt_text)

        # Get top chunks
        out_sims = []
        for emb,prompt in zip(prompt_embeddings, prompts):
            sims = cos_sim(emb, chunk_embeddings).cpu().numpy()
            out_sims.append(sims.squeeze())
            # Get diversity by randomly selecting context from highly similar documents
            mask = [s > MIN_COS_SIM for s in sims.squeeze()]
            usable_chunks = [chunks[k] for k in range(len(chunks)) if mask[k]]
            random.shuffle(usable_chunks)
            # Get the last chunks as well in case there aren't enough
            ranks = np.argsort(sims)[0,::-1]
            total = 0
            k = 0
            curr = []
            while total < 1000:
                if k >= len(usable_chunks):
                    chunk = chunks[ranks[k]]
                else:
                    chunk = usable_chunks[k]
                k += 1

                total += len(encoding.encode(chunk))
                curr.append(chunk)
            output_context[issue][prompt] = 'BEGIN CONTEXT=====\n' + encoding.decode(encoding.encode('\n=====\n'.join(curr))[:1000]) + '=====END CONTEXT'
        cos_sims[issue] = np.array(out_sims)
    print(f"Average token length: {np.mean(tok_lens)}")
    print(f"Median token length: {np.median(tok_lens)}")
    return output_context, cos_sims



if __name__ == '__main__':
    # Initialize the process group
    # dist.init_process_group(backend='nccl')

    # # Determine local rank and set device
    # local_rank = int(os.getenv('LOCAL_RANK', 0))
    # torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, help="Name of the output file",
                        required=True)
    parser.add_argument("--rag_file", type=str, help="Original file used for search",
                        required=True)
    parser.add_argument("--spec_file", type=str,
                        help="A json file with a list of models and topics to generate from", required=True)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)

    args = parser.parse_args()

    enforce_reproducibility(args.seed)
    seed = args.seed
    spec_file = args.spec_file
    output_file = Path(args.output_file)
    rag_file = args.rag_file

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
    # Pre-preform retrieval
    retrieved_contexts, cos_sims = perform_rag(rag_file, issues, prompts)

    with open(output_file, 'w') as f:
        f.write(json.dumps(retrieved_contexts))

    outdir = Path(output_file).parent
    np.savez(f"{outdir}/cos_sims.npz", **cos_sims)