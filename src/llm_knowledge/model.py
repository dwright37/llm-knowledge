import importlib.util

import torch
from torch import bfloat16
import transformers
from typing import List, Dict
from openai import OpenAI
#TODO: Commented until modelendpoints becomes public
#from modelendpoints.query import openai_batch
from tqdm import tqdm

if importlib.util.find_spec("vllm"):
    from vllm import LLM, SamplingParams

from .util.helpers import enforce_reproducibility


class QueryModel(object):
    """
    Abstract class for models
    """

    def __init__(
            self,
            model_id: str,
            seed: int = 1000
    ):
        """

        :param model_id: The model ID of the base model
        :param seed: Random seed for RNG
        """
        enforce_reproducibility(seed)
        self.model_id = model_id

    def generate(self, messages: List[List[Dict[str, str]]]) -> Dict[str, str]:
        """
        A function to generate outputs

        :param messages: List of list of dictionaries with each conversation we are generating
        :return: A dictionary of strings
        """
        pass


class OpenAIModel(QueryModel):
    def __init__(
            self,
            model_id: str,
            seed: int,
            openai_credentials: Dict[str, str],
            batched=False,
            **kwargs):
        """
        Query model with OAI models as the backend
        :param model_id: The OAI model we want to use
        :param seed: Seed for RNG
        :param openai_credentials: A dictionary with OAI credentials (api_key, organization, and project)
        :param kwargs:
        """
        super().__init__(model_id, seed)
        self._validate_credentials(openai_credentials)
        self.model_id = model_id
        self.client = OpenAI(**openai_credentials)
        self.batched = batched

    def _validate_credentials(self, openai_credentials: Dict[str, str]) -> None:
        """
        Validates the format of the OpenAI credentials
        :param openai_credentials: A dictionary with OAI credentials (api_key, organization, and project)
        """
        if openai_credentials is None:
            raise ValueError("OpenAI credentials should be provided")
        if "api_key" not in openai_credentials:
            raise ValueError("api_key should be provided")
        if "organization" not in openai_credentials:
            raise ValueError("organization should be provided")
        if "project" not in openai_credentials:
            raise ValueError("project should be provided")
        return True

    def _generate_single(
            self,
            messages: List[List[Dict[str, str]]],
            max_new_tokens: int = 4000,
            do_sample: bool = True,
            top_p: float = 0.9,
            temperature: float = 1.0,
            n_samples: int = 1,
            **kwargs
    ) -> Dict[str, str]:
        """
        A function to query the OpenAI client

        :param messages: List of list of dictionaries with each conversation we are generating
        :return: A dictionary with the output results (for now just contains "text" with a list of strings)
        """
        raw_results = []
        for msgs in tqdm(messages):
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=top_p,
                n=n_samples
            )
            raw_results.extend([completion.choices[i].message.content.strip() for i in
                                range(len(completion.choices))])
        return {
            "text": raw_results
        }

    def generate(
            self,
            messages: List[List[Dict[str, str]]],
            max_new_tokens: int = 4000,
            do_sample: bool = True,
            top_p: float = 0.9,
            temperature: float = 1.0,
            n_samples: int = 1,
            **kwargs
    ) -> Dict[str, str]:
        """
        A function to query the OpenAI client

        :param messages: List of list of dictionaries with each conversation we are generating
        :return: A dictionary with the output results (for now just contains "text" with a list of strings)
        """
        if self.batched:
            raise NotImplementedError
            # if self.model_id in ["gpt-5-2025-08-07"]:
            #     results = openai_batch(
            #         self.client,
            #         model=self.model_id,
            #         keys_to_messages={str(j): msgs for j, msgs in enumerate(messages)},
            #         max_completion_tokens=max_new_tokens,
            #         n=n_samples,
            #         reasoning_effort='minimal'
            #     )
            # else:
            #     results = openai_batch(
            #         self.client,
            #         model=self.model_id,
            #         keys_to_messages={str(j): msgs for j, msgs in enumerate(messages)},
            #         max_tokens=max_new_tokens,
            #         top_p=top_p,
            #         n=n_samples
            #     )
            # output = {
            #     'text': []
            # }
            # for id_ in results:
            #     output['text'].append(results[id_]['text'])
            # return output
        else:
            return self._generate_single(
                messages,
                max_new_tokens,
                do_sample,
                top_p,
                temperature,
                n_samples,
                **kwargs
            )


class HuggingfaceModel(QueryModel):
    def __init__(
            self,
            model_id: str,
            seed: int = 1000,
            quant: bool = True,
            **kwargs
    ):
        """
        Init a model with transformers backend
        :param model_id: Huggingface model ID of the model to load
        :param seed: Seed for RNG
        :param quant: Whether to use quantization
        :param kwargs:
        """
        super().__init__(model_id, seed)
        self.bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,  # 4-bit quantization
            bnb_4bit_quant_type="nf4",  # Normalized float 4
            bnb_4bit_use_double_quant=True,  # Second quantization after the first
            bnb_4bit_compute_dtype=bfloat16,  # Computation type
        )
        max_memory = {i: "30000MB" for i in range(torch.cuda.device_count())}
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            quantization_config=self.bnb_config if quant else None,
            device_map="auto",
            max_memory=max_memory,
        )
        self.model.eval()

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages for generation
        :param messages: List of list of dictionaries with each conversation we are generating
        :return: A string with the messages formatted for generation/completion
        """
        prompt = self.tokenizer.apply_chat_template(
            messages
        )
        return prompt

    def generate(
            self,
            messages: List[List[Dict[str, str]]],
            max_new_tokens: int = 4000,
            do_sample: bool = True,
            top_p: float = 0.9,
            temperature: float = 1.0,
            n_samples: int = 1,
            **kwargs
    ) -> Dict[str, str]:
        """
        Generate output from the underlying transformers model

        :param messages: List of list of dictionaries with each conversation we are generating
        :param max_new_tokens: Maximum number of new tokens for the model to generate
        :param do_sample: Whether to use greedy decoding or sample the output
        :param top_p: Fraction of probability distirbution to consider for sampling
        :param temperature: Output generation temperature
        :param n_samples: Number of random samples to generate for each stimulus
        :param kwargs:
        :return: A dictionary with the output results (for now just contains "text" with a list of strings)
        """
        pipeline = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature
        )
        raw_results = []
        for msgs in messages:
            prompt = self._format_messages(msgs)
            raw_result = pipeline(prompt)[0]["generated_text"]
            raw_results.append(raw_result.strip())
        return {
            "text": raw_results
        }


class VllmModel(QueryModel):
    def __init__(
            self,
            model_id: str,
            seed: int = 1000,
            n_gpu: int = -1,
            download_dir = "~/.cache/huggingface/hub/",
            max_model_len=None,
            gpu_memory_utilization=0.9,
            parallel=False,
            max_num_seqs=None,
            **kwargs
    ):
        """
        Init a model with vLLM backend
        :param model_id: Huggingface model ID of the model to load
        :param seed: Seed for RNG
        :param n_gpu: The number of GPUs to use. If -1, will use all available GPUs
        :param kwargs:
        """
        super().__init__(model_id, seed)
        if n_gpu == -1:
            n_gpu = torch.cuda.device_count()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        self.config = transformers.AutoConfig.from_pretrained(self.model_id)
        self.model = LLM(model=model_id, trust_remote_code=True,
                tensor_parallel_size=1 if parallel else n_gpu, pipeline_parallel_size=n_gpu if parallel else 1, dtype='auto', download_dir=download_dir,
                         max_model_len=min(max_model_len, self.config.max_position_embeddings if hasattr(self.config, 'max_position_embeddings') else 128000),
                         gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=max_num_seqs) #https://github.com/vllm-project/vllm/issues/5376

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages for generation
        :param messages: List of list of dictionaries with each conversation we are generating
        :return: A string with the messages formatted for generation/completion
        """
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_special_tokens=False,
            add_generation_prompt=True
        )
        if messages[-1]['role'] == 'assistant':
            prompt = prompt[:prompt.rfind(messages[-1]['content'].strip()) + len(messages[-1]['content'].strip())]
        return prompt

    def generate(
            self,
            messages: List[List[Dict[str, str]]],
            max_new_tokens: int = 4000,
            do_sample: bool = True,
            top_p: float = 0.9,
            temperature: float = 1.0,
            n_samples: int = 1,
            **kwargs
    ) -> Dict[str, str]:
        """
        Generate output from the underlying vLLM model

        :param messages: List of list of dictionaries with each conversation we are generating
        :param max_new_tokens: Maximum number of new tokens for the model to generate
        :param do_sample: Whether to use greedy decoding or sample the output
        :param top_p: Fraction of probability distirbution to consider for sampling
        :param temperature: Output generation temperature
        :param n_samples: Number of random samples to generate for each stimulus
        :param kwargs:
        :return: A dictionary with the output results. Contains "text" (list of strings containing the output),
        "tokens" (list of list of strings containing each token in the output), and "logprobs" (list of list of floats containing log probabilities of each token)
        """
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p,
            logprobs=1,
            n=n_samples
        )
        prompts = [self._format_messages(msgs) for msgs in messages]
        outputs = list(self.model.generate(prompts, sampling_params))
        text = [samp.text for o in outputs for samp in o.outputs]
        tokens = [[v.decoded_token for lp in samp.logprobs for k,v in list(sorted(lp.items(), key=lambda x: x[1].rank, reverse=True))[:1]] for o in outputs for samp in o.outputs]
        logprobs = [[v.logprob for lp in samp.logprobs for k,v in list(sorted(lp.items(), key=lambda x: x[1].rank, reverse=True))[:1]] for o in outputs for samp in o.outputs]
        return {
            "text": text,
            "tokens": tokens,
            "logprobs": logprobs
        }

    def unload(self):
        self.model.sleep(2)

    def reload(self):
        self.model.wake_up()


def get_model(
        model_id: str,
        seed: int = 1000,
        backend: str = 'vllm',
        n_gpu: int = -1,
        openai_credentials: Dict[str, str] = None,
        quant: bool = False,
        download_dir: str = None,
        max_model_len=128000,
        gpu_memory_utilization=0.9,
        parallel=False,
        max_num_seqs=None,
        batched=False
) -> QueryModel:
    """
    Return a model to use for analysis and inference
    :param model_id: The string ID of the model to load
    :param seed: Seed for RNG
    :param backend: The backend used to serve the model. Can be "openai", "transformers", or "vllm". Currently only "vllm" supports returning token probabilities.
    :param n_gpu: The number of GPUs to use. If -1, will use all available GPUs
    :param openai_credentials: A dictionary containing openai credentials if the openai backend us used (api_key, organization, and project).
    :param quant: Whether to quantize the model
    :return: A QueryModel object with the loaded model that can be used for inference.
    """
    assert backend in ['vllm', 'transformers', 'openai'], "Invalid model backend provided, please choose one of: 'vllm', 'transformers', 'openai'"
    if backend == 'openai':
        return OpenAIModel(
            model_id,
            seed,
            openai_credentials,
            batched=batched
        )
    elif backend == 'transformers':
        return HuggingfaceModel(
            model_id,
            seed,
            quant
        )
    else:
        return VllmModel(
            model_id,
            seed,
            n_gpu,
            download_dir,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            parallel=parallel,
            max_num_seqs=max_num_seqs
        )
