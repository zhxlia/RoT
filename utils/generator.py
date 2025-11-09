import os
import time
import openai
import torch

from tqdm import tqdm
from copy import deepcopy
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import List, Dict, Any, Tuple
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ProcessPoolExecutor


_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


class GPTOpenAIGenerator(object):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.error_types = {
            "continue_error": [
                "timed out",
                "Connection error",
                "Connection reset by peer",
                "Remote end closed connection without response",
                "occurred in violation of protocol",
                "Failed to resolve",
                "TLSV1_ALERT_INTERNAL_ERROR",
                "Error communicating",
                "The server is overloaded or not ready yet",
                "upstream_error",
                "new_api_error",
                "当前分组上游负载已饱和",
                "Lock wait timeout exceeded"
            ],
            "sleep_error": [
                "call rate limit",
                "token rate limit"
            ],
            "ignore_error": [
                "content",
                "reduce the length"
            ]
        }

    def generate_single(self, packed_data: List[tuple]) -> List[Tuple[str, float]]:
        openai.api_key = ""
        openai.api_base = ""

        sentence, engine, config = packed_data
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="",
                    messages=[{"role": "user", "content": sentence}],
                    **config
                )
                return [(c.message['content'].strip(), 1.0) for c in response.choices]
            except Exception as e:
                continue_flag = False
                sleep_flag = False
                ignore_flag = False
                for x in self.error_types['continue_error']:
                    if x in str(e):
                        continue_flag = True
                for x in self.error_types['sleep_error']:
                    if x in str(e):
                        sleep_flag = True
                        continue_flag = True
                for x in self.error_types['ignore_error']:
                    if x in str(e):
                        ignore_flag = True
                if sleep_flag:
                    time.sleep(5)
                if continue_flag:
                    continue
                if not ignore_flag:
                    print(e)
                return [("", 0.0)]

    def generate(self, source: List[str], config: Dict[str, Any]) -> List[List[Tuple[str, float]]]:
        config = deepcopy(config)
        if config['parallel']:
            config.pop('parallel')
            if 'batch_size' in config:
                config.pop('batch_size')
            packed_data = [(x, self.model_name, config) for x in source]
            with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as _:
                result: List[List[str]] = list(process_map(
                    self.generate_single, packed_data, max_workers=os.cpu_count() // 2, chunksize=1))
        else:
            config.pop('parallel')
            result: List[List[str]] = [self.generate_single(
                (x, self.model_name, config)) for x in tqdm(source)]
        return result


class VLLMGenerator(object):
    def __init__(self, model_name_or_path: str):
        def check_cuda_gt_8() -> bool:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                device_properties = torch.cuda.get_device_properties(i)
                compute_capability = float(
                    f"{device_properties.major}.{device_properties.minor}")
                if compute_capability < 8.0:
                    return False
            return True

        self.llm = LLM(model=model_name_or_path,
                       tensor_parallel_size=torch.cuda.device_count(),
                       dtype="auto" if check_cuda_gt_8() else "float",
                       trust_remote_code=True)
        self.tokenizer = self.llm.get_tokenizer()
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def _filter_too_long_input(self, source: List[str], config: Dict[str, Any]) -> List[str]:
        too_long_data_count = 0
        source_filtered = []
        for i, x in tqdm(enumerate(source), total=len(source), desc="filtering too long input"):
            if len(self.tokenizer(x)['input_ids']) > self.llm.llm_engine.model_config.max_model_len:
                source[i] = "TL;NR"
                too_long_data_count += 1
            else:
                source_filtered.append(x)
        print(f"too long input count: {too_long_data_count}")
        if config['ignore_too_long']:
            return source_filtered
        return source

    def generate(self, source: List[str], config: Dict[str, Any]) -> List[List[Tuple[str, float]]]:
        source = self._filter_too_long_input(source, config)
        sampling_params = SamplingParams(
            temperature=config['temperature'],
            top_p=config['top_p'],
            max_tokens=config['max_tokens'],
            n=config['n'],
            logprobs=1,
            stop=config['stop'] + [self.tokenizer.eos_token]
        )

        res_completions = []
        batch_size = config['batch_size']
        batch_instances = batch_data(
            source, batch_size=batch_size)
        for _, prompt in tqdm(enumerate(batch_instances), total=len(batch_instances), desc="generating"):
            completions = self.llm.generate(
                prompt, sampling_params, use_tqdm=False)
            for output in completions:
                prompt_results = []
                for out in output.outputs:
                    total_logprob = 0
                    for i, x in enumerate(out.logprobs):
                        token_idx = list(x.keys())[0]
                        if token_idx == self.pad_token_id or token_idx == self.eos_token_id:
                            break
                        total_logprob += x[token_idx].logprob
                    prompt_results.append((out.text, total_logprob))
                res_completions.append(prompt_results)

        return res_completions


class VLLMChatGenerator(VLLMGenerator):
    def generate(self, source: List[Dict[str, str]], config: Dict[str, Any]) -> List[List[Tuple[str, float]]]:
        messages_list = [[
            {"role": "user", "content": x['instruction']},
            {"role": "assistant", "content": x['response'] + _MAGIC_SPLITTER_}
        ] for x in source]
        source = [self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False) for messages in messages_list]
        source = [x.split(_MAGIC_SPLITTER_)[0] for x in source]
        print(source[0])
        return super().generate(source, config)


class VLLMLoRAGenerator(VLLMGenerator):
    def __init__(self, model_name_or_path: str, lora_name_or_path: str):
        def check_cuda_gt_8() -> bool:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                device_properties = torch.cuda.get_device_properties(i)
                compute_capability = float(
                    f"{device_properties.major}.{device_properties.minor}")
                if compute_capability < 8.0:
                    return False
            return True

        self.llm = LLM(model=model_name_or_path,
                       tensor_parallel_size=torch.cuda.device_count(),
                       dtype="auto" if check_cuda_gt_8() else "float",
                       trust_remote_code=True,
                       enable_lora=True,
                       max_lora_rank=64)
        self.adapter = LoRARequest(lora_name_or_path, 1, lora_name_or_path)
        self.tokenizer = self.llm.get_tokenizer()
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def generate(self, source: List[str], config: Dict[str, Any]) -> List[List[Tuple[str, float]]]:
        source = self._filter_too_long_input(source, config)
        sampling_params = SamplingParams(
            temperature=config['temperature'],
            top_p=config['top_p'],
            max_tokens=config['max_tokens'],
            n=config['n'],
            logprobs=1,
            stop=config['stop'] + [self.tokenizer.eos_token]
        )

        res_completions = []
        batch_size = config['batch_size']
        batch_instances = batch_data(
            source, batch_size=batch_size)
        for _, prompt in tqdm(enumerate(batch_instances), total=len(batch_instances), desc="generating"):
            completions = self.llm.generate(
                prompt, sampling_params, use_tqdm=False, lora_request=self.adapter)
            for output in completions:
                prompt_results = []
                for out in output.outputs:
                    total_logprob = 0
                    for i, x in enumerate(out.logprobs):
                        token_idx = list(x.keys())[0]
                        if token_idx == self.pad_token_id or token_idx == self.eos_token_id:
                            break
                        total_logprob += x[token_idx].logprob
                    prompt_results.append((out.text, total_logprob))
                res_completions.append(prompt_results)

        return res_completions


class VLLMLoRAChatGenerator(VLLMLoRAGenerator):
    def generate(self, source, config):
        messages_list = [[
            {"role": "user", "content": x['instruction']},
            {"role": "assistant", "content": x['response'] + _MAGIC_SPLITTER_}
        ] for x in source]
        source = [self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False) for messages in messages_list]
        source = [x.split(_MAGIC_SPLITTER_)[0] for x in source]
        print(source[0])
        return super().generate(source, config)


def batch_data(data_list: List[str], batch_size: int) -> List[List[str]]:
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1) * batch_size
        batch_data.append(data_list[start:end])
    last_start = (n-1) * batch_size
    batch_data.append(data_list[last_start:])
    return batch_data


def detect_generator(model_name_or_path: str, mode: str = 'chat', lora_name_or_path: str = None) -> object:
    if lora_name_or_path:
        if mode == 'chat':
            return VLLMLoRAChatGenerator(model_name_or_path, lora_name_or_path)
        return VLLMLoRAGenerator(model_name_or_path, lora_name_or_path)
    if 'gpt' in model_name_or_path:
        return GPTOpenAIGenerator(model_name_or_path)
    if mode == 'chat':
        return VLLMChatGenerator(model_name_or_path)
    return VLLMGenerator(model_name_or_path)


def generate_with_llm(model_name_or_path: str, source: List[Dict[str, str]], config: Dict[str, Any], mode: str = 'chat', lora_name_or_path: str = None) -> List[List[Tuple[str, float]]]:
    generator = detect_generator(model_name_or_path, mode, lora_name_or_path)
    if mode == 'text':
        results = generator.generate(
            [s['instruction'] for s in source], config)
    else:
        results = generator.generate(source, config)
    return results


def consistency(answers: List[Tuple[str, Any, float]]) -> Tuple[str, Any]:
    count: Dict[str, float] = {}
    record: Dict[str, Tuple[str, str]] = {}
    for a, b, c in answers:
        x = str(b)
        if "error" in x:
            continue
        if x not in count:
            count[x] = 0
            record[x] = (a, b)
        count[x] += c
    if not count:
        return "", ""
    return record[max(count, key=lambda x: count[x])]
