from typing import List, Tuple, Union
import uuid
import pandas as pd


from .model import QueryModel
from .util.helpers import enforce_reproducibility


def bulk_generate(
    stimuli: Union[List[str], List[Tuple[str, int]]],
    model: QueryModel,
    user_prompt_variations: List[str] = ["{proposition}"],
    system_prompt_variations: List[str] = [""],
    topic_ids: List[int] = None,
    seed: int = 1000,
    max_new_tokens: int = 4000,
    do_sample: bool = True,
    top_p: float = 0.9,
    temperatures: List[float] = [1.0],
    n_samples: int = 1,
    rag_context: List[str] = None
) -> pd.DataFrame:
    """
    Method to generate many responses from a given model for analysis
    :param stimuli: A list of propositions to use as inputs to the model or a tuple with a list of strings and integers for the stance valence
    :param model: A llm_knowledge.model.QueryModel object
    :param user_prompt_variations: A list of prompt variations which will be applied to every stimulus. Each string in the list should contain the string {proposition}
    :param system_prompt_variations: A list of system prompt variations (e.g., assigning personas to the model of the user)
    :param topic_ids: A list of integers the same size as stimuli indicating the topic of each stimulus
    :param seed: Seed for RNG
    :param max_new_tokens: Maximum number of new tokens for the model to generate
    :param do_sample: Whether to use greedy decoding or sample the output
    :param top_p: Fraction of probability distirbution to consider for sampling
    :param temperature: Output generation temperature
    :param n_samples: Number of random samples to generate for each stimulus
    :param rag_context: A list of strings to add to a vector database for RAG
    :return: A dataframe with len(stimuli)*len(user_prompt_variations)*len(system_prompt_variations)*n_samples rows and the following columns:
            -" stimulus_id": A unique ID for each stimulus provided in the "stimulus" list
            - "topic_id": The topic ID for the response
            - "stimulus": The stimulus used for this row
            - "system_prompt": The system prompt used to generate the response
            - "user_prompt": The user prompt used to generate the response
            - "text": The plain text of the response
            - "tokens": The individual tokens in the response
            - "logprobs": The log probability of each token in the response
    """

    enforce_reproducibility(seed)

    if isinstance(stimuli[0], Tuple):
        stimuli, stance_valence = zip(*stimuli)
    else:
        stance_valence = None
    # Create all of the prompt variations based on the system prompts, user prompts, and stimuli
    assert all("{proposition}" in p for p in user_prompt_variations), "All prompt variations must contain the replacements string '{proposition}'"
    stim_ids = [id_ for stim in stimuli for id_ in [str(uuid.uuid4())]*(len(user_prompt_variations)*len(system_prompt_variations)*n_samples)]
    stims = [st for stim in stimuli for st in [stim]*(len(user_prompt_variations)*len(system_prompt_variations)*n_samples)]

    if topic_ids != None:
        topic_ids = [id_ for topic_id in topic_ids for id_ in [topic_id]*(len(user_prompt_variations)*len(system_prompt_variations)*n_samples)]
    else:
        # Assume everything is a different topic
        topic_ids = list(range(len(stims)))
    assert len(stims) == len(stim_ids), f"{len(stims)}; {len(stim_ids)}"

    if rag_context is None:
        user_prompts = [p.replace("{proposition}", s) for s in stimuli for p in user_prompt_variations]
    else:
        # Rag context is keyed off of issue + prompt template
        user_prompts = [f'{p.replace("{proposition}", s)}\nThe following additional information may help in crafting your response: {rag_context[s][p]}'
                        for s in stimuli for p in user_prompt_variations]
    msgs = []
    
    for sp in system_prompt_variations:
        for up in user_prompts:
            assistant_prompt = [{"role": "user", "content": sp + '\n' + up}] if 'gemma' in model.model_id else [{"role": "system", "content": sp}, {"role": "user", "content": up}]
            msgs.append(assistant_prompt)

    all_dframes = []
    for temperature in temperatures:
        outputs = model.generate(
            msgs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            n_samples=n_samples
        )
        assert len(outputs['text']) == len(stim_ids)
        outputs['stimulus_id'] = stim_ids
        outputs['topic_id'] = topic_ids
        outputs['stimulus'] = stims
        outputs['temperature'] = [temperature]*len(stim_ids)
        if 'gemma' in model.model_id:
            outputs['system_prompt'] = ['' for m in msgs for c in [m[0]['content']] * n_samples]
            outputs['user_prompt'] = [c for m in msgs for c in [m[0]['content']] * n_samples]
        else:
            outputs['system_prompt'] = [c for m in msgs for c in [m[0]['content']]*n_samples]
            outputs['user_prompt'] =  [c for m in msgs for c in [m[1]['content']]*n_samples]
        if stance_valence is not None:
            outputs['stance_valence'] = [val for sv in stance_valence for val in [sv]*(len(user_prompt_variations)*len(system_prompt_variations)*n_samples)]
        all_dframes.append(pd.DataFrame.from_dict(outputs))

    return pd.concat(all_dframes, ignore_index=True)