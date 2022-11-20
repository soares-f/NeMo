# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import json

import torch
from tqdm.auto import tqdm

from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_learning_dataset import GPTPromptLearningDataset
from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids
from nemo.utils import logging

__all__ = ['GPTFinetuningDataset']


class GPTFinetuningDataset(GPTPromptLearningDataset):
    """
    The dataset class for prompt-tuning or p-tuning pretrained GPT models.
    """

    def __init__(
        self,
        datasets,
        tokenizer,
        task_templates,
        pad_token_id: str,
        max_seq_length: int,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        for_train: bool = True,
    ):
        self.tokenizer = tokenizer
        self.task_templates = task_templates
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.for_train = for_train
        self.examples = []

        assert self.min_seq_length <= max_seq_length, "Min sequence length should be less than or equal to max"
        assert self.max_seq_length > 0, "Max sequence length should be greater than 0"

        logging.info("Loading and tokenizing dataset ... ")

        # Datasets is just a list of json dicts
        if isinstance(datasets[0], dict):
            self.load_data(datasets)

        # Datasets are a list of file path strings to .json or .jsonl files
        elif isinstance(datasets[0], str):
            for path in datasets:
                dataset = open(path, 'r', encoding='utf-8')
                self.load_data(dataset)
        else:
            raise ValueError("Datasets must be a list of dicts or a list of filepath strings")

    def load_data(self, dataset):
        """
        Loads a dataset by filling in the task templates specified in the config file
        with the information from each training/inference example. Converts all input 
        text into token ids. Also replaces the <|VIRTUAL_PROMPT_#|> placeholders in 
        the task templates with the actual virtual prompt token ids. 
        params:
            dataset: A list of json objects or a dictionary objects each
                     containing the information needed for a training example
        """
        skipped = 0

        for json_line in tqdm(dataset):

            # Read example dict or load the information for a single example from .json file
            if type(json_line) == dict:
                doc = json_line
            else:
                doc = json.loads(json_line)

            taskname = doc["taskname"]
            prompt_template = self.task_templates[taskname]["prompt_template"]
            prompt_template_fields = self.task_templates[taskname]["prompt_template_fields"]
            truncation_field = self.task_templates[taskname]['truncate_field']
            answer_only_loss = self.task_templates[taskname]['answer_only_loss']
            answer_field = self.task_templates[taskname]["answer_field"]

            input_example = prompt_template

            self._input_sanity_checks(
                prompt_template,
                prompt_template_fields,
                truncation_field,
                answer_only_loss,
                answer_field,
                doc,
            )

            # Format the input example according to the template
            input_example = self._insert_text_in_template(input_example, prompt_template_fields, doc)
            input_ids = self.tokenizer.text_to_ids(input_example)

            # Add BOS/EOS if desired, adds EOS by default
            if self.add_bos:
                input_ids = [self.tokenizer.bos_id] + input_ids
            if self.add_eos:
                input_ids = input_ids + [self.tokenizer.eos_id]

            # Try to truncate input text to fit into the max sequence length
            if len(input_ids) > self.max_seq_length:
                input_ids = self._truncate_input(truncation_field, input_ids, taskname, doc)

            # Skip example if the final length doesn't fit length requirements even after truncation
            if self.min_seq_length <= len(input_ids) <= self.max_seq_length:

                # Find answer field indices if training and answer_only_loss is True
                answer_start_idx = None
                if answer_only_loss and self.for_train:
                    answer_start_idx = self._find_answer_start(taskname, input_ids, answer_field, doc)

                self.examples.append((input_ids, answer_start_idx))
            else:
                skipped += 1

        logging.info(f'Skipped {skipped} sentences, sequence length too short or too long even after truncation')

    def _input_sanity_checks(
        self,
        prompt_template,
        prompt_template_fields,
        truncation_field,
        answer_only_loss,
        answer_field,
        doc,
    ):   
        # Check if input example has fields not present in template
        keys_not_in_template = list(set(doc.keys()) - set(prompt_template_fields) - set(['taskname']))
        assert (
            len(keys_not_in_template) == 0
        ), f"Examples in your dataset contain the fields: {keys_not_in_template} that are not in the task template."

        # Check that answer field checks if answer_only_loss was set to True
        if answer_only_loss and self.for_train:
            assert answer_field is not None, "If answer_only_loss=True, an answer_field must be given"
            assert (
                answer_field in doc.keys()
            ), f"answer_only_loss=True but the given answer_field '{answer_field}' is not in data json"
            assert truncation_field != answer_field, "Answer field and truncation field should not match"

            answer_placeholder = "{" + answer_field + "}"
            answer_placeholder_len = len(answer_placeholder)
            placeholder_start = len(prompt_template) - answer_placeholder_len
            assert prompt_template[placeholder_start:] == answer_placeholder, "Answer field must be at prompt end"

    def collate_fn(self, batch):
        """ Prepares input_ids, labels, loss mask, attention_mask, and position ids for global batch """
        # Get max sequence length of batch
        input_ids, answer_starts = zip(*batch)
        batch_max = max(len(ids) for ids in input_ids)
        input_ids, loss_mask = self.pad_batch_and_build_loss_mask(input_ids, batch_max, answer_starts)

        # Should be a label for every token in batch, label is the next token
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        batch_max -= 1

        # Loss mask should align with labels
        loss_mask = loss_mask[:, 1:].contiguous()

        # Using causal attention mask for whole input
        batch_size = len(input_ids)
        attention_mask = torch.tril(torch.ones((batch_size, batch_max, batch_max))).view(
            batch_size, 1, batch_max, batch_max
        )

        # Convert attention mask from float to bool
        attention_mask = attention_mask < 0.5
        position_ids = build_position_ids(input_ids)

        return input_ids, labels, loss_mask, attention_mask, position_ids

    def get_all_examples(self, tokens_to_generate):
        """
        Used for loading inference data. 
        """
        input_ids, answer_starts = zip(*self.examples)
        input_lengths = torch.cuda.LongTensor([len(inputs) for inputs in input_ids])
        batch_max = input_lengths.max().item()
        batch_max += tokens_to_generate

        input_ids, _ = self.pad_batch_and_build_loss_mask(input_ids, batch_max, answer_starts)
        input_ids = input_ids.cuda()
        input_ids = torch.cuda.LongTensor(input_ids)

        return input_ids, input_lengths