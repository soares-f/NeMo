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

import re
import torch

from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.collections.nlp.data.language_modeling.megatron.gpt_finetuning_dataset import GPTFinetuningDataset
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

try:
    from apex.transformer import parallel_state
    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


class MegatronGPTFineTuneModel(MegatronGPTModel):
    """
    Megatron GPT Finetuning of a Pretrained Model
    """   
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        self.load_task_templates(self.cfg.task_templates)
        self.pad_token_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else self.tokenizer.unk_id


    def load_task_templates(self, task_templates):
        """
        Takes in the task template portion of the config and turns  
        it into a table where each task's prompt template and 
        the number of virtual tokens to insert in a given part of 
        the prompt template are specified. 
        """
        self.task_templates = {}
        for task in task_templates:
            self.task_templates[task.taskname] = {
                "prompt_template": task.prompt_template,
                "prompt_template_fields": re.findall("\{(.*?)\}", task.prompt_template),
                "answer_only_loss": task.get("answer_only_loss", True), # set answer only loss
                "answer_field": task.get("answer_field", None),
                "truncate_field": task.truncate_field,
            }

    def training_step(self, batch, batch_idx):
        """
            Dataloader produces a global batch which is turned into a list of microbatches.
            The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """

        # we zero grads here because we also call backward in the apex fwd/bwd functions
        self._optimizer.zero_grad()

        # we prepare the micro batches for the apex fwd/bwd function
        loss_mean = self.fwd_bwd_step(batch, forward_only=False)
        self.reduce_gradients(loss_mean)
        self.log_training_step(loss_mean)

        return loss_mean

    def validation_step(self, batch, batch_idx):
        loss_mean = self.fwd_bwd_step(batch, forward_only=True)
        if loss_mean.item == 0.0:
            loss_mean = []

        return loss_mean

    def validation_epoch_end(self, outputs):
        if not outputs:
            return
        if parallel_state.is_pipeline_last_stage():
            # only the last pipeline parallel stages return loss
            averaged_loss = torch.stack(outputs).mean()
        else:
            averaged_loss = torch.tensor(0.0).cuda()

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(averaged_loss, get_last_rank())

        self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True)

    def setup(self, stage=None):
        """ PTL hook that is executed after DDP spawns.
            We setup datasets here as megatron datasets require DDP to instantiate.
            See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.
        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """
        #self.setup_consumed_samples()

        if stage == 'predict':
            return
        else:
            self.setup_training_data(self.cfg.data)
            self.setup_validation_data(self.cfg.data)
            self.setup_test_data(self.cfg.data)

        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            self.model.sync_initial_word_embeddings()

    def setup_training_data(self, cfg):
        if self.cfg.data.get('train_ds', None):
            self._train_ds, self._train_dl = self.build_finetuning_dataset(
                dataset_paths=self.cfg.data.train_ds,
                batch_size=self.cfg.global_batch_size,
                for_train=True,
                drop_last=True,
                shuffle=True,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

    def setup_validation_data(self, cfg):
        if self.cfg.data.get('validation_ds', None):
            self._validation_ds, self._validation_dl = self.build_finetuning_dataset(
                dataset_paths=self.cfg.data.validation_ds,
                batch_size=self.cfg.global_batch_size,
                for_train=True,
                drop_last=True,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

    def setup_test_data(self, cfg):
        if self.cfg.data.get('test_ds', None):
            self._test_ds, self._test_dl = self.build_finetuning_dataset(
                dataset_paths=self.cfg.data.test_ds,
                batch_size=self.cfg.global_batch_size,
                for_train=False,
                drop_last=False,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

    def build_finetuning_dataset(
        self, dataset_paths, batch_size, for_train, drop_last, shuffle, num_workers, pin_memory
    ):
        dataset = GPTFinetuningDataset(
            datasets=dataset_paths,
            tokenizer=self.tokenizer,
            task_templates=self.task_templates,
            pad_token_id=self.pad_token_id,
            max_seq_length=self.cfg.data.get('max_seq_length', self.cfg.max_position_embeddings),
            min_seq_length=self.cfg.data.get('min_seq_length', 1),
            add_bos=self.cfg.data.get('add_bos', False),
            add_eos=self.cfg.data.get('add_eos', True),
            for_train=for_train,
        )

        rank = parallel_state.get_data_parallel_rank()
        world_size = parallel_state.get_data_parallel_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return dataset, dataloader

    def list_available_models(self):
        return None