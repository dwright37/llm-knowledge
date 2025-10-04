<div align="center">
  

# 🧐`llm_knowledge`🧐

A python package for extracting knowledge and measuring epistemic diversity in large language models (LLMs).

Associated with the papers "Epistemic Diversity and Knowledge Collapse in Large Language Models" (Wright et al. 2025) and "LLM Tropes: Revealing Fine Grained Values and Opinions in Large Language Models" [(Wright et al. 2024)](https://aclanthology.org/2024.findings-emnlp.995/)

</div>

## Installation

The package can be installed in the following ways:

1) CPU only (note this will prevent usage of vLLM):

```bash
$ git clone git@github.com:dwright37/llm-knowledge.git
$ cd llm-knowledge
$ pip install .
```

2) With GPU support:

```bash
$ git clone git@github.com:dwright37/llm-knowledge.git
$ cd llm-knowledge
$ pip install .[gpu]
```

## Usage

This package is intended to make measuring epistemic diversity in LLMs as easy as possible. This involves three primary functions:

1) Sampling outputs from a model given a set of prompts
2) Partitioning these outputs into decomposed claims of equivalent meaning
3) Measuring coverage and epistemic diversity of the sample taken

### Extracting Claim Clusters

Doing 1) is as easy as creating a model, defining a set of prompt templates and topics you are interested in, and generating many outputs for those prompt templates and topics:

```python
from llm_knowledge.model import get_model
from llm_knowledge.generate import bulk_generate

# Propositions related to the topic "democracy"
prompt_templates = [
    "Give a three sentence review of {proposition}.", 
    "Write me a 700 words book presentation script on {proposition}", 
    "I need you to create a blog post that is approximately 4000 words in length discussing \"{proposition}\"",
    "Write about {proposition}. Write at least 5 paragraphs, please."
]

topics = ["democracy", "feminism"]

# Create a model
model = get_model('meta-llama/Meta-Llama-3.1-8B-Instruct', seed=1000, backend='vllm')

# Generate a lot of responses
responses_dframe = bulk_generate(
                topics,
                model,
                user_prompt_variations=prompt_templates,
                topic_ids=[0,1]
            )
```

Doing 2) is done by first decomposing:

```python
from llm_knowledge.epistemic_diversity import extract_claims_bulk

# Extract epistemic_diversity and their probabilities of occurring
factoid_dframe = extract_claims_bulk(
                responses_dframe,
                model,
                group_key='topic'
            )
```

and then clustering:

```python
from llm_knowledge.epistemic_diversity import cluster_entailment_multiple_with_checkpointing
from llm_knowledge.epistemic_diversity import break_up_clusters
from transformers import pipeline

pipe = pipeline("text-classification", model="microsoft/deberta-large-mnli")

out_dframe = cluster_entailment_multiple_with_checkpointing(
    pipe,
    factoid_dframe,
    outfile_name="clusters.pqt",
    checkpoint_steps=50000,
    N=6
)

out_dframe = break_up_clusters(
    pipe,
    out_dframe,
    outfile_name="clusters.pqt",
    checkpoint_steps=50000,
    N=6
)
```

This will result in a parquet file `clusters.pqt` where each row is a claim and with the following columns:

- `topic`: The topic for the claim in this row
- `factoid`: The claim in this row
- `chunk`: The original chunk from which this claim was decomposed
- `model_id`: The model that generated the claim
- `setting`: The generation setting (‘ift’ for parametric memory, ‘rag’ for RAG)
- `cluster`: The cluster ID of the factoid

### Measuring Epistemic Divrsity

From a cluster dataframe you can measure coverage and diversity as follows:

```python
from llm_knowledge.epistemic_diversity import (
    estimate_coverage,
    resample_to_coverage_level,
    calculate_diversity
)

# Estimates the coverage for a specific model and setting
coverage_level = estimate_coverage(
    out_dframe, 
    group_keys=['model_id', 'setting'], 
    group_values=[model_id, 'ift']
)

sample_dframe = resample_to_coverage_level(
    out_dframe, 
    group_keys=['model_id', 'setting'], 
    group_values=[model_id, 'ift'], 
    coverage_level=min_coverage_level # Select a minimum coverage level to rarefy the sample
)

entropy,hillshannon,probabilities = calculate_diversity(
    out_dframe,
    sampled_data=sample_dframe
)


```

## Examples

Examples from our epistemic diversity paper can be be found under `src/experiments`. These examples demonstrate the three primary usages of this package: sampling many outputs from a set of propositions, 
extracting and clustering claims from these outputs, and measuring epistemic diversity.

## Knowledge Collapse Experiments

### Important data

- Most of the data in under `experiments/data`
- The final clusters will be released soon
- Prompt templates are given at `experiments/data/issuebench_templates.json`
- Model information is given in `experiments/data/model_categories.json`
- All topics are listed in `experiments/data/topics.txt`
- All of the prompts for the various parts of the pipeline (local decomposition, wikipedia decomposition, and LLM as a judge) are under `experiments/data/prompts`

### Important code

- Backbone code for decomposition and clustering is under `src/epistemic_diversity.py`; you can install the package locally by running `$ pip install -e .[gpu]`
- Code for running the different parts of the pipeline are under `experiments/generate.py`, `experiments/decompose.py`, `experiments/cluster_batched.py`, and `experiments/break_up_large_clusters.py`
- Code for G-Eval is under `experiments/LLM_judge`
- Code for generating the plots in the paper is in `experiments/final_plots.ipynb`

# Citation
The code in this package is derived from our recent preprint and our [EMNLP Findings 2024 paper](https://aclanthology.org/2024.findings-emnlp.995/):

```
@article{wright2025epistemicdiversity,
      title={Epistemic Diversity and Knowledge Collapse in Large Language Models},
      author={Dustin Wright and Sarah Masud and Jared Moore and Srishti Yadav
                and Maria Antoniak and Chan Young Park and Isabelle Augenstein},
      year={2025},
      journal={arXiv preprint arXiv:TODO},
}
```

```
@inproceedings{wright2024revealingfinegrainedvaluesopinions,
      title={LLM Tropes: Revealing Fine-Grained Values and Opinions in Large Language Models},
      author={Dustin Wright and Arnav Arora and Nadav Borenstein and Srishti Yadav and Serge Belongie and Isabelle Augenstein},
      year={2024},
      booktitle = {Findings of EMNLP},
      publisher = {Association for Computational Linguistics}
}
```
