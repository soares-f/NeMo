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

from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.plugins.environments.torchelastic_environment import TorchElasticEnvironment

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_finetune_model import (
    MegatronGPTFineTuneModel,
)
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager


# From https://github.com/NVIDIA/NeMo/tree/finetune-gpt
"""
This is an example of how to finetune a GPT model using simple supervised
learning. This will take a dataset in the same format of adapter tuning
and perform updates using only the answer part of the whole prompt.
"""


@hydra_runner(config_path="conf", config_name="megatron_gpt_finetuning")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )
    if cfg.trainer.precision == 16:
        scaler = GradScaler(
            init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
            growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
            hysteresis=cfg.model.get('hysteresis', 2),
        )
        plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)
    exp_manager(trainer, cfg.exp_manager)

    # Override timer callback to a stateless one
    for idx, callback in enumerate(trainer.callbacks):
        if isinstance(callback, Timer):
            trainer.callbacks[idx] = StatelessTimer(cfg.trainer.max_time,)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    gpt_model_cfg = MegatronGPTModel.restore_from(
            cfg.model.get('restore_path'), trainer=trainer, return_config=True
        )

    # Need to overwrite some params in gpt model's config before restoring
    with open_dict(gpt_model_cfg):
        gpt_model_cfg.megatron_amp_O2 = False
        gpt_model_cfg.micro_batch_size = cfg.model.micro_batch_size
        gpt_model_cfg.global_batch_size = cfg.model.global_batch_size
        gpt_model_cfg.precision = trainer.precision
        gpt_model_cfg.data.data_prefix = None
        gpt_model_cfg.optim = cfg.model.optim
        gpt_model_cfg.data.train_ds = cfg.model.data.train_ds
        gpt_model_cfg.data.validation_ds = cfg.model.data.validation_ds
        gpt_model_cfg.task_templates = cfg.model.task_templates



    # load existing or init new soft prompt GPT model
    model = MegatronGPTFineTuneModel.restore_from(
        cfg.model.restore_path, override_config_path=gpt_model_cfg, trainer=trainer, save_restore_connector=NLPSaveRestoreConnector()
    )

    trainer.fit(model)


if __name__ == '__main__':
    main()