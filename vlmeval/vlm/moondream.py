import torch
import re
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
import copy
import safetensors
import torch
import torch.nn as nn

from contextlib import contextmanager
from typing import Callable, List

def extract_object(sentence: str) -> str:
    words = sentence.split()
    obj_words = []
    for word in words[2:]:
        if word.lower() in {"are", "is"}:
            break
        obj_words.append(word)
        
    return " ".join(obj_words)

class Moondream1(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path="vikhyatk/moondream1", **kwargs):
        try:
            from transformers import (
                AutoModelForCausalLM,
                CodeGenTokenizerFast as Tokenizer,
            )
        except Exception as e:
            logging.critical(
                "Please install Transformers version 4.36.2 by running: 'pip install transformers==4.36.2', "
                "please intall torchvision>=0.16."
            )
            raise e

        assert osp.exists(model_path) or splitlen(model_path) == 2

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        self.tokenizer = Tokenizer.from_pretrained(model_path)

        default_kwargs = dict(max_new_tokens=512)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs

        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        prompt, img = self.message_to_promptimg(message)
        enc_image = self.model.encode_image(Image.open(img))

        prompt_wtmpl = f"<image>\n\nQuestion: {prompt}\n\nAnswer:"
        answer = self.model.generate(
            enc_image,
            prompt_wtmpl,
            eos_text="<END>",
            tokenizer=self.tokenizer,
            **self.kwargs,
        )[0]
        cleaned_answer = re.sub("<$", "", re.sub("END$", "", answer)).strip()
        return cleaned_answer

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(["MMMU"], dataset):
            return False
        if DATASET_TYPE(dataset) == "MCQ" or dataset in [
            "MMVet",
        ]:
            return True

        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        question = line["question"]
        if dataset == "MMVet":
            prompt = question + "\nAnswer the question directly. "
        elif DATASET_TYPE(dataset) == "MCQ":
            options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
            options_prompt = ""
            for key, item in options.items():
                options_prompt += f"{key}. {item}\n"

            hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
            prompt = f"Hint: {hint}\n" if hint is not None else ""
            prompt += f"{question}\n"
            prompt += (
                f"{options_prompt}\nAnswer with the option’s letter from the given choices directly. "
                if len(options)
                else "Answer the question directly. "
            )
        else:
            raise NotImplementedError

        message = [dict(type="text", value=prompt)]
        message.extend([dict(type="image", value=s) for s in tgt_path])
        return message


class Moondream2(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False
    def __init__(self, model_path="vikhyatk/moondream2", revision=None, local_path=None, **kwargs):
        try:
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if transformers.__version__ != '4.44.0':
                raise ImportError("Transformers 4.44.0 required")
        except ImportError:
            logging.critical("Install transformers==4.44.0 and torchvision>=0.16")
            raise
                    
        assert osp.exists(model_path) or splitlen(model_path) == 2

        self.model = AutoModelForCausalLM.from_pretrained(
            "moondream/moondream-2b-2025-04-14-4bit",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": "cuda"},
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.kwargs = {"max_new_tokens": 512, **kwargs}
        
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
        torch.cuda.empty_cache()
        

    def generate_inner(self, message, dataset=None, countbench_mode="point"):
        """
        Generate an answer for the given message and dataset.
        Args:
            message (dict): The message containing the question and image.
            dataset (str): The dataset for which the answer is being generated.
            countbench_mode (str): The mode for CountbenchQA, either "point" or "query".
        Returns:
            str: The generated answer.
        """
        prompt, img = self.message_to_promptimg(message)
        enc_image = self.model.encode_image(Image.open(img))
        # print(f"Prompt for {dataset} -> ", prompt)
        
        if dataset == "CountbenchQA" and countbench_mode == "point":
            return len(self.model.point(enc_image, prompt)["points"])
        
        return self.model.query(enc_image, prompt)["answer"].strip()

    def use_custom_prompt(self, dataset):
        """
        Determine if a custom prompt is needed for the given dataset.
        Args:
            dataset (str): The dataset for which the prompt is being checked.
        Returns:
            bool: True if a custom prompt is needed, False otherwise.
        """
        assert dataset is not None

        if listinstr(["MMMU"], dataset):
            return False
        elif DATASET_TYPE(dataset) == "MCQ":
            return True
        elif dataset in [
            "ChartQA_TEST",
            "TextVQA_VAL",
            "DocVQA_VAL",
            "POPE",
            "RealWorldQA",
            "TallyQA",
            "CountbenchQA",
            "MMVet",
        ]:
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        
        tgt_path = self.dump_image(line, dataset)
        question = line["question"]

        countbench_mode="point" # Change this to "query" for CountbenchQA query mode. 
        
        prompts = {
            "ChartQA_TEST": f"Analyze the chart carefully, consider both visual features and data values, and provide a precise answer without any additional explanation or formatting. {question}",
            "TextVQA_VAL": f"Read the text in the image and provide a brief lowercase answer. Respond 'unanswerable' only if there is no plausible answer. {question}",
            "DocVQA_VAL": f"{question} The answer should be a short text span taken verbatim from the document.",
            "POPE": f"{question}\nAnswer yes or no.",
            "TallyQA": f"Look at the image carefully and count the objects. Answer with just a number, without any additional text. {question}",
            "CountbenchQA_query": f"Look at the image carefully and count the objects. Answer with just a number, without any additional text. {question}",
            "CountbenchQA_point": f"individual {extract_object(question)}",
            "RealWorldQA": f"Look at the image carefully and answer the question. Provide a brief answer without any additional text. {question}",
            "MMVet": f"{question}\nAnswer the question directly."
        }
        
        if dataset == "CountbenchQA":
            prompt_key = f"CountbenchQA_{countbench_mode}"
            prompt = prompts[prompt_key]
        elif dataset in prompts:
            prompt = prompts[dataset]
        elif DATASET_TYPE(dataset) == "MCQ":
            options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
            options_prompt = ""
            for key, item in options.items():
                options_prompt += f"{key}. {item}\n"

            hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
            prompt = f"Hint: {hint}\n" if hint is not None else ""
            prompt += f"{question}\n"
            prompt += (
                f"{options_prompt}\nAnswer with the option’s letter from the given choices directly. "
                if len(options)
                else "Answer the question directly. "
            )
        else:
            raise NotImplementedError
        
        return [{"type": "text", "value": prompt}] + [{"type": "image", "value": s} for s in tgt_path]
