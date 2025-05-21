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


@contextmanager
def safetensors_open(safetensors_file: str):
    """
    Simplify interfacing with safetensors files. Eliminates the need to ignore
    type errors when using the `safe_open` function.
    """
    with safetensors.safe_open(
        safetensors_file, framework="pt"
    ) as st:  # pyright: ignore

        def get_tensor(name: str) -> torch.Tensor:
            return st.get_tensor(name)

        def get_keys() -> List[str]:
            return st.keys()

        get_tensor.keys = get_keys

        yield get_tensor


# def _load_weights(get_tensor: Callable[[str], torch.Tensor], model: nn.Module) -> None:
#     """Internal function to load weights using a tensor getter function."""
#     model = model.to(dtype=torch.float16)

#     vision = model.vision
#     region = model.region
#     weight_map = {
#         "vision_encoder.encoder.model.visual.patch_embed.linear.weight": vision[
#             "patch_emb"
#         ].weight,
#         "vision_encoder.encoder.model.visual.patch_embed.linear.bias": vision[
#             "patch_emb"
#         ].bias,
#         "vision_encoder.encoder.model.visual.pos_embed": vision.pos_emb,
#         "vision_encoder.encoder.model.visual.norm.weight": vision["post_ln"].weight,
#         "vision_encoder.encoder.model.visual.norm.bias": vision["post_ln"].bias,
#         "vision_encoder.projection.mlp.fc1.weight": vision["proj_mlp"]["fc1"].weight,
#         "vision_encoder.projection.mlp.fc1.bias": vision["proj_mlp"]["fc1"].bias,
#         "vision_encoder.projection.mlp.fc2.weight": vision["proj_mlp"]["fc2"].weight,
#         "vision_encoder.projection.mlp.fc2.bias": vision["proj_mlp"]["fc2"].bias,
#         "text_model.transformer.embd.wte.weight": model.text.wte,
#         "text_model.lm_head.ln.weight": model.text["post_ln"].weight,
#         "text_model.lm_head.ln.bias": model.text["post_ln"].bias,
#         "text_model.lm_head.linear.weight": model.text["lm_head"].weight,
#         "text_model.lm_head.linear.bias": model.text["lm_head"].bias,
#         "region_model.coordinate_encoder.weight": region["coord_encoder"].weight,
#         "region_model.coordinate_encoder.bias": region["coord_encoder"].bias,
#         "region_model.coordinate_decoder.fc1.weight": region["coord_decoder"][
#             "fc1"
#         ].weight,
#         "region_model.coordinate_decoder.fc1.bias": region["coord_decoder"]["fc1"].bias,
#         "region_model.coordinate_decoder.fc2.weight": region["coord_decoder"][
#             "fc2"
#         ].weight,
#         "region_model.coordinate_decoder.fc2.bias": region["coord_decoder"]["fc2"].bias,
#         "region_model.size_encoder.weight": region["size_encoder"].weight,
#         "region_model.size_encoder.bias": region["size_encoder"].bias,
#         "region_model.size_decoder.fc1.weight": region["size_decoder"]["fc1"].weight,
#         "region_model.size_decoder.fc1.bias": region["size_decoder"]["fc1"].bias,
#         "region_model.size_decoder.fc2.weight": region["size_decoder"]["fc2"].weight,
#         "region_model.size_decoder.fc2.bias": region["size_decoder"]["fc2"].bias,
#     }

#     for i in range(len(model.vision["blocks"])):
#         prefix = f"vision_encoder.encoder.model.visual.blocks.{i}"
#         blk = model.vision["blocks"][i]
#         weight_map.update(
#             {
#                 f"{prefix}.norm1.weight": blk["ln1"].weight,
#                 f"{prefix}.norm1.bias": blk["ln1"].bias,
#                 f"{prefix}.norm2.weight": blk["ln2"].weight,
#                 f"{prefix}.norm2.bias": blk["ln2"].bias,
#                 f"{prefix}.attn.qkv.weight": blk["attn"]["qkv"].weight,
#                 f"{prefix}.attn.qkv.bias": blk["attn"]["qkv"].bias,
#                 f"{prefix}.attn.proj.weight": blk["attn"]["proj"].weight,
#                 f"{prefix}.attn.proj.bias": blk["attn"]["proj"].bias,
#                 f"{prefix}.mlp.fc1.weight": blk["mlp"]["fc1"].weight,
#                 f"{prefix}.mlp.fc1.bias": blk["mlp"]["fc1"].bias,
#                 f"{prefix}.mlp.fc2.weight": blk["mlp"]["fc2"].weight,
#                 f"{prefix}.mlp.fc2.bias": blk["mlp"]["fc2"].bias,
#             }
#         )

#     for i in range(len(model.text["blocks"])):
#         prefix = f"text_model.transformer.h.{i}"
#         blk = model.text["blocks"][i]
#         weight_map.update(
#             {
#                 f"{prefix}.ln.weight": blk["ln"].weight,
#                 f"{prefix}.ln.bias": blk["ln"].bias,
#                 f"{prefix}.mixer.Wqkv.weight": blk["attn"]["qkv"].weight,
#                 f"{prefix}.mixer.Wqkv.bias": blk["attn"]["qkv"].bias,
#                 f"{prefix}.mixer.out_proj.weight": blk["attn"]["proj"].weight,
#                 f"{prefix}.mixer.out_proj.bias": blk["attn"]["proj"].bias,
#                 f"{prefix}.mlp.fc1.weight": blk["mlp"]["fc1"].weight,
#                 f"{prefix}.mlp.fc1.bias": blk["mlp"]["fc1"].bias,
#                 f"{prefix}.mlp.fc2.weight": blk["mlp"]["fc2"].weight,
#                 f"{prefix}.mlp.fc2.bias": blk["mlp"]["fc2"].bias,
#             }
#         )

#     for key, tensor in weight_map.items():
#         tensor.data.copy_(get_tensor(key))

#     region.coord_features.data.copy_(
#         get_tensor("region_model.coordinate_features.weight").T
#     )
#     region.size_features.data.copy_(get_tensor("region_model.size_features.weight").T)


# def load_weights_from_safetensors(weights_file: str, model: nn.Module) -> None:
#     """Load weights from a safetensors file into a MoondreamModel instance."""
#     with safetensors_open(weights_file) as get_tensor:
#         if (
#             "vision.blocks.0.attn.proj.bias" in get_tensor.keys()
#             or "model.vision.blocks.0.attn.proj.bias" in get_tensor.keys()
#         ):
#             with safetensors_open(weights_file) as get_tensor:
#                 tensors = {
#                     k.replace("model.", ""): get_tensor(k) for k in get_tensor.keys()
#                 }
#                 model.load_state_dict(tensors, strict=False)
#         else:
#             # Wrap the get_tensor function to handle key normalization
#             name_map = {k.replace("._orig_mod", ""): k for k in get_tensor.keys()}
#             _load_weights(
#                 lambda x: get_tensor(name_map[x]).to(dtype=torch.float16), model
#             )


# def load_weights_from_pt(weights_file: str, model: nn.Module) -> None:
#     """Load weights from a PyTorch file into a MoondreamModel instance."""
#     device = str(torch.empty(0).device)
#     tensors = torch.load(weights_file, map_location=device, weights_only=True)
#     if "vision.blocks.0.attn.proj.bias" in tensors.keys():
#         model.load_state_dict(tensors, strict=False)
#     else:
#         tensors = {
#             k.replace("._orig_mod", ""): v.to(dtype=torch.float16)
#             for k, v in tensors.items()
#         }
#         _load_weights(lambda x: tensors[x], model)


# def load_weights_into_model(weights_file: str, model: nn.Module) -> None:
#     """
#     Load weights from either a safetensors or PyTorch file directly into a MoondreamModel instance.

#     Args:
#         weights_file: Path to weights file (either .safetensors or .pt)
#         model: MoondreamModel instance to load weights into
#     """
#     if weights_file.endswith(".safetensors"):
#         load_weights_from_safetensors(weights_file, model)
#     else:
#         load_weights_from_pt(weights_file, model)

#     # Make all parameters contiguous
#     for param in model.parameters():
#         param.data = param.data.contiguous()

def extract_object(sentence: str) -> str:
    # Split the sentence into words.
    words = sentence.split()
    
    # Starting from the third word (index 2)
    obj_words = []
    for word in words[2:]:
        # Stop if we encounter "are" or "is" (case-insensitive)
        if word.lower() in {"are", "is"}:
            break
        obj_words.append(word)
        
    # Join and return the collected words as a string.
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
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            logging.critical(
                """Please install Transformers version 4.44 by running: "pip install transformers==4.44.0",
            please intall torchvision>=0.16."""
            )
            raise e

        assert osp.exists(model_path) or splitlen(model_path) == 2

        self.model = AutoModelForCausalLM.from_pretrained(
            "moondream/moondream-2b-2025-04-14-4bit",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": "cuda"},
            # revision=revision,
        )


        
        # from .config import MoondreamConfig
        # from .moondream import MoondreamModel
        # from .weights import load_weights_into_model
        # config = MoondreamConfig()
        # self.model = MoondreamModel(config) 
        # load_weights_into_model(local_path, self.model)       

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        default_kwargs = dict(max_new_tokens=512)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs

        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        prompt, img = self.message_to_promptimg(message)
        enc_image = self.model.encode_image(Image.open(img))
        print(f"prompt for {dataset} -> ", prompt)

        if dataset == "CountbenchQA":
            answer = self.model.query(enc_image, prompt)["answer"]
            #answer = len(self.model.point(enc_image, prompt)["points"])
            #cleaned_answer = answer#answer.strip()
            cleaned_answer = answer
        else:
            answer = self.model.query(enc_image, prompt)["answer"]
            cleaned_answer = answer.strip()
            
        return cleaned_answer

    def use_custom_prompt(self, dataset):
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

        if dataset == "ChartQA_TEST":
            prompt = (
                "Analyze the chart carefully, consider both visual features and data values,"
                " and provide a precise answer without any additional explanation or formatting. "
                + question
            )
        elif dataset == "TextVQA_VAL":
            prompt = (
                "Read the text in the image and provide a brief lowercase answer. "
                "Respond 'unanswerable' only if there is no plausible answer. "
                + question
            )
        elif dataset == "DocVQA_VAL":
            prompt = question + " The answer should be a short text span taken verbatim from the document."
        elif dataset == "POPE":
            prompt = f"{question}\nAnswer yes or no."
        # elif dataset == "RealWorldQA":
        #     prompt = question
        elif dataset == "TallyQA":# or dataset == "CountbenchQA":
            prompt = (
                "Look at the image carefully and count the objects. "
                "Answer with just a number, without any additional text. "
                + question
            )
        elif dataset == "CountbenchQA":
            prompt = (
                "Look at the image carefully and count the objects. "
                "Answer with just a number, without any additional text. "
                + question
            )
            #prompt = "individual " + extract_object(question)

        elif dataset == "MMVet":
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

