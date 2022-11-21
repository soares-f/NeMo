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
from functools import partial

from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.collections.nlp.data.language_modeling.megatron.gpt_finetuning_dataset import GPTFinetuningDataset

from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_learning_dataset import GPTPromptLearningDataset
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

from nemo.collections.nlp.modules.common import (
    BIGLSTMPromptEncoder,
    VirtualPromptPlaceholderToken,
    VirtualPromptSource,
    VirtualPromptStyle,
)


try:
    from apex.transformer import parallel_state, tensor_parallel
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
        forward_backward_pipelining_without_interleaving,
    )
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import forward_backward_no_pipelining

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
        self.pipeline_parallel = self.cfg.get('pipeline_model_parallel_size', 1) > 1


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
                "total_virtual_tokens" : 0,
                "virtual_token_splits": task.virtual_token_splits,
            }

    def set_inference_config(self, inference_config):
        self._inference_config = inference_config

    def get_inference_config(self):
        return self._inference_config

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.
        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""

        self.model.set_input_tensor(input_tensor)

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(batch, model):
            batch = [x.cuda(non_blocking=True) for x in batch]
            input_ids, labels, loss_mask, position_ids, attention_mask, taskname_ids = batch
            output_tensor = model(input_ids, position_ids, attention_mask, labels)

            if isinstance(output_tensor, tuple):
                output_tensor, _ = output_tensor

            def loss_func(output_tensor):
                loss = model.loss_func(loss_mask, output_tensor)
                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def get_forward_output_only_func(self):
        """
        Used for generate method only for now.
        """

        def fwd_output_only_func(batch, model):
            extra_arg = {}
            (
                tokens,
                attention_mask,
                position_ids,
                task_ids,
                set_inference_key_value_memory,
                inference_max_sequence_len,
            ) = batch

            tokens = tokens.cuda()
            attention_mask = attention_mask.cuda()
            position_ids = position_ids.cuda()
            task_ids = task_ids.cuda()
            extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
            extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()

            output_tensor = model(tokens, position_ids, attention_mask, task_ids, **extra_arg)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def on_train_end(self):
        # Save the best nemo model
        self.save_to(save_path=self.cfg.nemo_path)

    def training_step(self, batch, batch_idx):
        """
            Dataloader produces a global batch which is turned into a list of microbatches.
            The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """

        # we zero grads here because we also call backward in the apex fwd/bwd functions
        self._optimizer.zero_grad()

        # we prepare the micro batches for the apex fwd/bwd function
        loss_mean = self.fwd_bwd_step(batch, batch_idx, forward_only=False)
        self.allreduce_gradients()

        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.cfg.precision == 16:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True)
        self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True)
        
        return loss_mean

    def log_training_step(self, loss_mean):
        if self.cfg.precision == 16:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True)
        self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True)
        # TODO: make sure compute_consumed_samples works for pipeline parallelism
        # self.log(
        #     'consumed_samples',
        #     self.compute_consumed_samples(self.trainer.global_step - self.init_global_step),
        #     prog_bar=True,
        #     rank_zero_only=True,
        # )
        
    def fwd_bwd_step(self, batch, batch_idx, forward_only):
        """
            Dataloader produces a global batch which is turned into a list of microbatches.
            The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """
        # Get seq length of batch
        _, seq_length = batch[0].shape
        tensor_shape = [seq_length, self.cfg.micro_batch_size, self.hidden_size]

        if self.pipeline_parallel:
            losses_reduced_per_micro_batch = forward_backward_pipelining_without_interleaving(
                forward_step_func=self.get_forward_output_and_loss_func(),
                batch=batch,
                model=self,
                forward_only=forward_only,
                tensor_shape=tensor_shape,
                dtype=self.autocast_dtype,
                grad_scaler=self.trainer.precision_plugin.scaler if self.cfg.precision == 16 else None,
                sequence_parallel_enabled=self.cfg.get("sequence_parallel", False),
            )
        else:
            losses_reduced_per_micro_batch = forward_backward_no_pipelining(
                forward_step_func=self.get_forward_output_and_loss_func(),
                batch=batch,
                model=self,
                forward_only=forward_only,
                tensor_shape=tensor_shape,
                dtype=self.autocast_dtype,
                grad_scaler=self.trainer.precision_plugin.scaler if self.cfg.precision == 16 else None,
            )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # average loss across micro batches
            loss_tensors_list = [loss_reduced['avg'] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.concat(loss_tensors_list)
            loss_mean = loss_tensor.mean()
        else:
            # we're not on the last pipeline stage so no losses
            loss_mean = torch.tensor(0.0).cuda()

        return loss_mean

    def validation_step(self, batch, batch_idx):
        loss_mean = self.fwd_bwd_step(batch, batch_idx, forward_only=True)
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
                data=self.cfg.data.train_ds,
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
                data=self.cfg.data.validation_ds,
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
                data=self.cfg.data.test_ds,
                batch_size=self.cfg.global_batch_size,
                for_train=False,
                drop_last=False,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
            )

    # def build_finetuning_dataset(
    #     self, dataset_paths, batch_size, for_train, drop_last, shuffle, num_workers, pin_memory
    # ):
    #     dataset = GPTFinetuningDataset(
    #         datasets=dataset_paths,
    #         tokenizer=self.tokenizer,
    #         task_templates=self.task_templates,
    #         pad_token_id=self.pad_token_id,
    #         max_seq_length=self.cfg.data.get('max_seq_length', self.cfg.max_position_embeddings),
    #         min_seq_length=self.cfg.data.get('min_seq_length', 1),
    #         add_bos=self.cfg.data.get('add_bos', False),
    #         add_eos=self.cfg.data.get('add_eos', True),
    #         for_train=for_train,
    #     )

    #     rank = parallel_state.get_data_parallel_rank()
    #     world_size = parallel_state.get_data_parallel_world_size()
    #     sampler = torch.utils.data.distributed.DistributedSampler(
    #         dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
    #     )

    #     dataloader = torch.utils.data.DataLoader(
    #         dataset,
    #         collate_fn=dataset.collate_fn,
    #         sampler=sampler,
    #         batch_size=batch_size,
    #         drop_last=drop_last,
    #         num_workers=num_workers,
    #         pin_memory=pin_memory,
    #     )

    #     return dataset, dataloader


    def build_finetuning_dataset(
        self,
        data,
        batch_size=None,
        max_seq_length=2048,
        min_seq_length=1,
        add_bos=False,
        add_eos=False,
        for_train=True,
        drop_last=False,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        tokens_to_generate=None,
        get_dataset_only=False,
        cache_data_path=None,
        load_cache=False,
    ):
        dataset = GPTPromptLearningDataset(
            data=data,
            tokenizer=self.tokenizer,
            virtual_prompt_source=VirtualPromptSource.NO_PROMPT,
            task_templates=self.task_templates,
            pseudo_tokens=[],
            pad_token_id=self.pad_token_id,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            add_bos=add_bos,
            add_eos=add_eos,
            for_train=for_train,
            tokens_to_generate=tokens_to_generate,
            cache_data_path=cache_data_path,
            load_cache=load_cache,
        )

        if get_dataset_only:
            return dataset

        # Make distributed dataloader
        rank = parallel_state.get_data_parallel_rank()
        data_parallel_size = parallel_state.get_data_parallel_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=data_parallel_size, rank=rank, shuffle=shuffle
        )

        assert batch_size % data_parallel_size == 0, "Global batch size must be evenly divisible by data parallel size"

        if for_train:
            if self.cfg.get("sequence_parallel", False):
                collate_fn = partial(
                    dataset.collate_fn, tp_workers=parallel_state.get_tensor_model_parallel_world_size()
                )
            else:
                collate_fn = partial(dataset.collate_fn, tp_workers=0)
        else:
            collate_fn = dataset.inference_collate_fn

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            sampler=sampler,
            batch_size=batch_size // data_parallel_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return dataset, dataloader

    def list_available_models(self):
        return None