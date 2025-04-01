# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on transformers/src/transformers/models/llama/modeling_llama.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch InternLMXComposer2 model."""
import os
import re
import copy
import queue
import threading
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from PIL import Image
import numpy as np
import random
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers.modeling_outputs import CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.utils import (add_start_docstrings_to_model_forward,
                                replace_return_docstrings)
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
try:
    from transformers.generation.streamers import BaseStreamer
except:  # noqa # pylint: disable=bare-except
    BaseStreamer = None

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from .build_mlp import build_vision_projector, build_vision_tower
from .ixc_utils import Image_transform, Video_transform, load_video, frame2img, get_font
from .configuration_internlm_xcomposer2 import InternLMXcomposer2Config
from .modeling_internlm2 import (InternLM2_INPUTS_DOCSTRING, InternLM2Model,
                                 InternLM2PreTrainedModel)

_CONFIG_FOR_DOC = 'InternLMXcomposer2Config'

image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv'}


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


def get_stopping_criteria(stop_words_ids):
    stop_words_ids = [torch.tensor([i]).cuda() for i in stop_words_ids]
    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids)])
    return stopping_criteria


def set_random_seed(seed, set_cudnn=False):
    """Set the random seed for reproducibility.

    Parameters:
    seed (int): The seed to use for generating random numbers.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # For multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    if set_cudnn and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def find_subarray_indices(tensor, subarray):
    tensor_len = len(tensor)
    subarray_len = len(subarray)
    indices = []

    if subarray_len > tensor_len:
        return indices  # Subarray longer than tensor, can't be a match

    for i in range(tensor_len - subarray_len + 1):
        if torch.equal(tensor[i:i + subarray_len], subarray):
            indices.append((i, i + subarray_len))

    return indices


class InternLMXComposer2ForCausalLM(InternLM2PreTrainedModel):
    _auto_class = 'AutoModelForCausalLM'

    _tied_weights_keys = ['output.weight']

    def __init__(self, config):
        super().__init__(config)
        self.model = InternLM2Model(config)
        self.vocab_size = config.vocab_size

        self.score = nn.Linear(config.hidden_size, 1, bias=False)
        
        self.tokenizer = None
        self.hd_num = 25
        self.font = get_font()

        self.max_length = config.max_length
        print(f'Set max length to {self.max_length}')
        # Initialize weights and apply final processing
        self.post_init()
        self.plora_glb_GN = nn.Parameter(torch.zeros([1, 1, 4096]))
        self.plora_sub_GN = nn.Parameter(torch.zeros([1, 1, 1, 4096]))

        self.vit = build_vision_tower()
        self.vision_proj = build_vision_projector()

        self.vis_processor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])


    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, InternLM2Model):
            module.gradient_checkpointing = value
        if value:
            self.vit.vision_tower.vision_model.encoder.gradient_checkpointing = value

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def encode_text(self, text, add_special_tokens=False):
        token = self.tokenizer(
            text, return_tensors='pt',
            add_special_tokens=add_special_tokens).input_ids.to(self.device)
        embs = self.model.tok_embeddings(token)
        return embs

    def encode_img(self, image, hd_num=25):
        if image is None:
            return None
        if isinstance(image, str):
            _, ext = os.path.splitext(image)
            if ext.lower() in image_extensions:
                image = Image.open(image).convert('RGB')
                image = Image_transform(image, hd_num = hd_num)
            elif ext.lower() in video_extensions:
                image = load_video(image)
                image = frame2img(image, self.font)
                image = Video_transform(image, hd_num = hd_num)
            else:
                print ('Unknow input format', image)
                return None
            image = self.vis_processor(image).unsqueeze(0).to(self.device)
        else:
            assert isinstance(image, torch.Tensor)

        img_embeds, atts_img, img_target = self.img2emb(image)
        return img_embeds

    def img2emb(self, image):
        img_embeds, img_split = self.vit([image], 
            self.plora_glb_GN, self.plora_sub_GN)
        if len(img_split) > 1:
            print ('Batch Size >1 is not supported.')
            assert 0
        #print (img_embeds.shape)
        img_embeds = self.vision_proj(img_embeds)
        atts_img = torch.ones(
            img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)

        img_target = torch.ones(
            img_embeds.size()[:2], dtype=torch.long).to(
                img_embeds.device) * -100

        return img_embeds, atts_img, img_target

    def prompt_wrap(self, img_embeds, prompt):
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.tokenizer(
            p_before, return_tensors='pt',
            add_special_tokens=True).to(img_embeds.device)

        p_before_embeds = self.model.tok_embeddings(
            p_before_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds], dim=1)

        wrapped_atts_img = torch.ones(
            wrapped_img_embeds.size()[:-1],
            dtype=torch.long).to(img_embeds.device)

        wrapped_target = torch.ones(
            batch_size, wrapped_img_embeds.shape[1], dtype=torch.long).to(
                img_embeds.device) * -100

        return wrapped_img_embeds, wrapped_atts_img, wrapped_target

    def text2emb(self, text, add_special_tokens=False):
        to_regress_tokens = self.tokenizer(
            text,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=add_special_tokens
        ).to(self.device)

        targets = self.mask_human_targets(to_regress_tokens.input_ids)
        targets = targets.to(self.device)
        return to_regress_tokens, targets

    def apply_chat_template(self, conversation, image, max_length: int=16384, hd_num: int=24, apply_template=True):
        if apply_template:
            prompt = ''
            for message in conversation:
                role = message['role']
                content = message['content']
                if role in ['system', 'user', 'assistant']:
                    prompt += f"""[UNUSED_TOKEN_146]{role}\n{content}[UNUSED_TOKEN_145]\n"""
                else:
                    raise NotImplementedError(f"The role '{role}' is not a valid")

            # end
            prompt = prompt + '</s>'
            # reward token id
            prompt = prompt + '[UNUSED_TOKEN_130]'
        else:
            image_nums = len(image)
            prompt = conversation

        image_nums = len(image)
        if image_nums == 1 and prompt.find('<ImageHere>') == -1:
            # print ('auto append image at the begining')
            prompt = '<ImageHere>' + prompt

        parts = prompt.split('<ImageHere>')
        wrap_tokens = []
        wrap_embeds, wrap_im_mask = [], []
        temp_len = 0
        need_bos = True

        if len(parts) != image_nums + 1:
            #raise ValueError('Invalid <ImageHere> prompt format.')
            print ('Waring! The image number != given position!')
        if image_nums > 1:
            hd_num = 6
        else:
            hu_num = hd_num
        for idx, part in enumerate(parts):
            if need_bos or len(part) > 0:
                part_tokens = self.tokenizer(
                    part,
                    return_tensors='pt',
                    padding='longest',
                    add_special_tokens=need_bos).to(self.device)
                if need_bos:
                    need_bos = False

                wrap_tokens.append(part_tokens.input_ids)

                part_embeds = self.model.tok_embeddings(
                    part_tokens.input_ids)
                wrap_embeds.append(part_embeds)
                wrap_im_mask.append(torch.zeros(part_embeds.shape[:2]))
                temp_len += part_embeds.shape[1]
            if idx < image_nums:
                if isinstance(image[idx], str):
                    img = self.encode_img(image[idx], hd_num)
                else:
                    # torch.tensor
                    img, _, _ = self.img2emb(image[idx])
                wrap_embeds.append(img)
                wrap_token = torch.ones(img.shape[:2], dtype=torch.long).to(self.device) * -100
                wrap_tokens.append(wrap_token)
                wrap_im_mask.append(torch.ones(img.shape[:2]))
                temp_len += img.shape[1]
            if temp_len > max_length:
                break

        wrap_tokens = torch.cat(wrap_tokens, dim=1)
    
        wrap_embeds = torch.cat(wrap_embeds, dim=1)
        wrap_im_mask = torch.cat(wrap_im_mask, dim=1)
        wrap_embeds = wrap_embeds[:, :max_length].to(self.device)
        wrap_im_mask = wrap_im_mask[:, :max_length].to(self.device).bool()
        return wrap_embeds, wrap_im_mask, temp_len

    def get_score(self, conversation: List[dict], image: List[str], max_length: int=16384, hd_num: int=24, apply_template: bool=True):
        inputs_embeds, im_mask, _ = self.apply_chat_template(conversation, image, max_length, hd_num, apply_template)
        attention_mask = torch.ones(1, inputs_embeds.shape[1]).to(bool).to(self.device)
        outputs = self.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, im_mask=im_mask)
        score = outputs.logits.cpu().item()
        return score

    def get_scores(self, conversations: List[List[dict]], images: List[List[str]], max_length: int=16384, hd_num: int=24, apply_template: bool=True):
        temp_embeds = []
        temp_im_mask = []
        for conversation, image in zip(conversations, images):
            inputs_embeds, im_mask, _ = self.apply_chat_template(conversation, image, max_length, hd_num, apply_template)
            temp_embeds.append(inputs_embeds)
            temp_im_mask.append(im_mask)

        temp_max_len = np.max([i.shape[1] for i in temp_embeds])
        temp_max_len = min(temp_max_len, max_length)

        batch_input_embeds, batch_atts, batch_im_mask = [], [], []
        pad = torch.ones([1, 1]) * self.tokenizer.pad_token_id
        pad = pad.long().to(self.device)
        pad_emb = self.model.tok_embeddings(pad)

        for idx in range(len(temp_embeds)):
            temp_len = temp_embeds[idx].shape[1]
            dtype = temp_im_mask[idx].dtype
            if temp_len >= temp_max_len:
                batch_input_embeds.append(temp_embeds[idx][:, :temp_max_len])
                batch_atts.append(torch.ones(1, temp_max_len).to(dtype).to(self.device))
                batch_im_mask.append(temp_im_mask[idx][:, :temp_max_len])
            else:
                batch_input_embeds.append(torch.cat([temp_embeds[idx], pad_emb.repeat(1, temp_max_len-temp_len, 1)], dim=1))
                batch_atts.append(torch.cat([torch.ones(1, temp_len), torch.zeros(1, temp_max_len-temp_len)], dim=1).to(dtype).to(self.device))
                batch_im_mask.append(torch.cat([temp_im_mask[idx], (torch.zeros(1, temp_max_len-temp_len)).to(dtype).to(self.device)], dim=1))

        batch_inputs_embeds = torch.cat(batch_input_embeds, dim=0)
        batch_atts = torch.cat(batch_atts, dim=0)
        batch_im_mask = torch.cat(batch_im_mask, dim=0)

        outputs = self.forward(inputs_embeds=batch_inputs_embeds, attention_mask=batch_atts, im_mask=batch_im_mask)
        scores = outputs.logits.squeeze().cpu().tolist()
        return scores

    @torch.no_grad()
    def compare(self, conversation1: List[dict], image1: List[str], conversation2: List[dict], image2: List[str], max_length: int=16384, hd_num: int=24, return_logits: bool=False, apply_template: bool=True):
        score1 = self.get_score(conversation1, image1, max_length, hd_num, apply_template)
        score2 = self.get_score(conversation2, image2, max_length, hd_num, apply_template)
        if return_logits:
            return score1 > score2, [score1, score2]
        else:
            return score1 > score2

    @torch.no_grad()
    def rank(self, conversations: List[List[dict]], images: List[List[str]], max_length: int=16384, hd_num: int=24, return_logits: bool=False, apply_template: bool=True):
        scores = self.get_scores(conversations, images, max_length, hd_num, apply_template)
        if return_logits:
            return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True), scores
        else:
            return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    def interleav_wrap(self, img_list, text_list, image_nums):
        temp_tokens = []
        temp_embeds = []
        temp_im_mask = []
        temp_tars = []

        # encode_image
        img_embeds, img_split = self.vit(img_list, self.plora_glb_GN, self.plora_sub_GN)
        img_embeds = self.vision_proj(img_embeds)

        for idx, text in enumerate(text_list):
            idx_ = idx // 2
            image_num = image_nums[idx_]
            im_id = int(np.sum(image_nums[:idx_]))
            images = []
            for i in range(image_num):
                st = int(np.sum(img_split[:im_id + i]))
                sp = img_split[im_id + i]
                temp_img = img_embeds[:, st:st+sp]
                images.append(temp_img)
            atts_img = torch.ones((len(images), images[0].shape[1]), dtype=torch.long).to(self.device)
            img_target = torch.ones(
                (len(images), images[0].shape[1]), dtype=torch.long).to(
                    self.device) * -100

            if image_num == 1 and text.find('<ImageHere>') == -1:
                text = '<ImageHere>' + text
            parts = text.split('<ImageHere>')

            wrap_tokens, wrap_embeds, wrap_im_mask = [], [], []
            temp_len = 0
            need_bos = True
            for idx, part in enumerate(parts):
                if need_bos or len(part) > 0:
                    part_tokens = self.tokenizer(part, return_tensors='pt', padding='longest',
                                                 add_special_tokens=need_bos).to(self.device)
                    if need_bos:
                        need_bos = False
                    wrap_tokens.append(part_tokens.input_ids)
                    part_embeds = self.model.tok_embeddings(part_tokens.input_ids)
                    wrap_embeds.append(part_embeds)
                    wrap_im_mask.append(torch.zeros(part_embeds.shape[:2]).to(self.device))
                    temp_len += part_embeds.shape[1]
                if idx < image_num:
                    wrap_embeds.append(images[idx])
                    wrap_token = torch.ones(images[idx].shape[:2], dtype=torch.long).to(self.device) * -100
                    wrap_tokens.append(wrap_token)
                    wrap_im_mask.append(torch.ones(images[idx].shape[:2]).to(self.device))
                    temp_len += images[idx].shape[1]
                if temp_len > self.max_length:
                    break
            wrap_tokens = torch.cat(wrap_tokens, dim=1)
            wrap_embeds = torch.cat(wrap_embeds, dim=1)
            wrap_im_mask = torch.cat(wrap_im_mask, dim=1)

            wrap_target = self.mask_human_targets(wrap_tokens).to(self.device)

            temp_tokens.append(wrap_tokens)
            temp_embeds.append(wrap_embeds)
            temp_im_mask.append(wrap_im_mask)
            temp_tars.append(wrap_target)

        temp_max_len = np.max([i.shape[1] for i in temp_embeds])
        temp_max_len = min(temp_max_len, self.max_length)

        final_input_ids, final_input_embeds, final_atts, final_tars, final_mask = [], [], [], [], []
        pad = torch.ones([1, 1]) * self.tokenizer.pad_token_id
        pad = pad.long().to(self.device)
        pad_emb = self.model.tok_embeddings(pad)

        for idx in range(len(temp_embeds)):
            temp_len = temp_embeds[idx].shape[1]
            if temp_len >= temp_max_len:
                final_input_ids.append(temp_tokens[idx][:, :temp_max_len])
                final_input_embeds.append(temp_embeds[idx][:, :temp_max_len])
                final_atts.append(torch.ones(1, temp_max_len).to(wrap_target.dtype).to(self.device))
                final_tars.append(temp_tars[idx][:, :temp_max_len])
                final_mask.append(temp_im_mask[idx][:, :temp_max_len])
            else:
                final_input_ids.append(torch.cat([temp_tokens[idx], (torch.ones(1, temp_max_len-temp_len) * self.tokenizer.pad_token_id).to(wrap_target.dtype).to(self.device)], dim=1))
                final_input_embeds.append(torch.cat([temp_embeds[idx], pad_emb.repeat(1, temp_max_len-temp_len, 1)], dim=1))
                final_atts.append(torch.cat([torch.ones(1, temp_len), torch.zeros(1, temp_max_len-temp_len)], dim=1).to(wrap_target.dtype).to(self.device))
                final_tars.append(torch.cat([temp_tars[idx], (torch.ones(1, temp_max_len-temp_len)*-100).to(wrap_target.dtype).to(self.device)], dim=1))
                final_mask.append(torch.cat([temp_im_mask[idx], (torch.zeros(1, temp_max_len-temp_len)).to(wrap_target.dtype).to(self.device)], dim=1))

        input_ids = torch.cat(final_input_ids, dim=0)
        inputs_embeds = torch.cat(final_input_embeds, dim=0)
        attention_mask = torch.cat(final_atts, dim=0)
        targets = torch.cat(final_tars, dim=0)
        im_mask = torch.cat(final_mask, dim=0)

        # to avoid error in DPO loss
        input_ids[input_ids == -100] = self.tokenizer.pad_token_id

        return input_ids, inputs_embeds, attention_mask, targets, im_mask

    def mask_human_targets(self, input_ids, pure=False):
        target_batch = []
        system_tokens = torch.tensor([92543, 9081]).to(self.device)
        for bs in range(input_ids.shape[0]):
            ids = input_ids[bs]
            targets = copy.deepcopy(ids)
            end_count = 0
            last_eoa = 0
            # 92542 -> [UNUSED_TOKEN_145]
            # 92543 -> [UNUSED_TOKEN_146]
            # 9081 -> system
            for i, temp_id in enumerate(ids):
                if temp_id == 92542:
                    search_results = find_subarray_indices(targets[last_eoa:i + 1], system_tokens)
                    if len(search_results) > 0:
                        targets[last_eoa:i + 1] = -100
                        last_eoa = i + 1
                    else:
                        if end_count % 2 == 0:
                            targets[last_eoa:i + 6] = -100
                        else:
                            last_eoa = i + 1
                        end_count += 1
                # # eos and following pad
                elif temp_id == 2:
                    # loss on eos, but not on pad
                    targets[i + 1:] = -100
                    break
            # trunction, end at last question
            if temp_id != 2 and end_count % 2 == 0:
                # mask all after the last answer
                targets[last_eoa + 1:] = -100
            target_batch.append(targets.unsqueeze(0))
        target_batch = torch.cat(target_batch, dim=0)
        return target_batch

    @add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """

        samples = kwargs.get('samples', None)
        if samples:
            infer_mode = samples.get('infer_mode', 'base')
            if samples['data_type'][0] == 'text':
                has_img = False
            elif samples['data_type'][0] == 'multi':
                has_img = True
            else:
                raise NotImplementedError

            # encode text
            text_chosen = samples['chosen'][0]
            text_rejected = samples['rejected'][0]

            text = [x for pair in zip(text_chosen, text_rejected) for x in pair]

            # encode image
            if has_img:
                image = samples['image'][0]
                bs = len(text)
                image_nums = []
                temp_image = []
                for im in image:
                    if type(im) is list:
                        image_nums.append(len(im))
                        temp_image.extend(im)
                    else:
                        image_nums.append(1)
                        temp_image.append(im)
                image = temp_image

                assert type(image) is list and len(image_nums) * 2 == bs

                input_ids_for_loss, to_regress_embeds, attention_mask, targets, im_mask = self.interleav_wrap(
                    image, text, image_nums)
            else:
                to_regress_tokens, targets = self.text2emb(
                    text, add_special_tokens=True)
                to_regress_embeds = self.model.tok_embeddings(
                    to_regress_tokens.input_ids)
                attention_mask = to_regress_tokens.attention_mask
                im_mask = torch.zeros(to_regress_embeds.shape[:2]).cuda()
                input_ids_for_loss = to_regress_tokens.input_ids

            input_ids_for_loss = input_ids_for_loss[:, :self.max_length]
            inputs_embeds = to_regress_embeds[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]
            targets = targets[:, :self.max_length]
            im_mask = im_mask[:, :self.max_length].bool()
            labels = targets
        else:
            im_mask = kwargs.get('im_mask', None)
            infer_mode = kwargs.get('infer_mode', 'base')
            if im_mask is None and inputs_embeds is not None:
                im_mask = torch.zeros(inputs_embeds.shape[:2]).to(
                    inputs_embeds.device)
                im_mask = im_mask.bool()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            im_mask=im_mask,
            infer_mode=infer_mode,
        )

        hidden_states = transformer_outputs[0]

        logits = self.score(hidden_states)
        logits = logits.float()

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1,1)
        pooled_logits = torch.gather(logits.squeeze(-1), 1, ends)

        loss = None
        if self.training:
            chosen_idx = torch.arange(0, batch_size, 2)
            rejected_idx = chosen_idx + 1
            loss = -F.logsigmoid(pooled_logits[chosen_idx] - pooled_logits[rejected_idx]).mean()

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
            

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      attention_mask=None,
                                      inputs_embeds=None,
                                      im_mask=None,
                                      infer_mode='base',
                                      **kwargs):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}

        im_mask = im_mask

        model_inputs.update({
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
            'im_mask': im_mask,
            'infer_mode': infer_mode, 
        })
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past), )
        return reordered_past

    def build_inputs(self,
                     tokenizer,
                     query: str,
                     history: List[Tuple[str, str]] = [],
                     meta_instruction=''):
        prompt = ''
        if meta_instruction:
            prompt += f"""<s>[UNUSED_TOKEN_146]system\n{meta_instruction}[UNUSED_TOKEN_145]\n"""
        else:
            prompt += '<s>'
        for record in history:
            prompt += f"""[UNUSED_TOKEN_146]user\n{record[0]}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{record[1]}[UNUSED_TOKEN_145]\n"""
        prompt += f"""[UNUSED_TOKEN_146]user\n{query}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""
        return tokenizer([prompt], return_tensors='pt')
