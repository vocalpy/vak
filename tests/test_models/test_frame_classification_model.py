import copy
from .test_definition import TweetyNetDefinition


import pytest
import torch

import vak.models

class TestFrameClassificationModel:

    MODEL_DEFINITION_MAP = {
        'TweetyNet': TweetyNetDefinition,
    }

    @pytest.mark.parametrize(
        'model_name',
        [
            'TweetyNet',
        ]
    )
    def test_load_state_dict_from_path(self,
                                       model_name,
                                       specific_config_toml_path,
                                       device
                                       ):
        """Smoke test that makes sure ``load_state_dict_from_path`` runs without failure.

        We use actual model definitions here so we can test with real checkpoints.
        """
        definition = self.MODEL_DEFINITION_MAP[model_name]
        train_toml_path = specific_config_toml_path('train', model_name, audio_format='cbin', annot_format='notmat')
        train_cfg = vak.config.Config.from_toml_path(train_toml_path)

        # stuff we need just to be able to instantiate network
        labelmap = vak.common.labels.to_map(train_cfg.prep.labelset, map_background=True)
        train_dataset = vak.datapipes.frame_classification.TrainDatapipe.from_dataset_path(
            dataset_path=train_cfg.train.dataset.path,
            split="train",
            window_size=train_cfg.train.dataset.params['window_size'],
        )
        input_shape = train_dataset.shape
        num_input_channels = input_shape[-3]
        num_freqbins = input_shape[-2]

        # network is the one thing that has required args
        # and we also need to use its config from the toml file
        cfg = vak.config.Config.from_toml_path(train_toml_path)
        model_config = cfg.train.model.asdict()
        network = definition.network(num_classes=len(labelmap),
                                     num_input_channels=num_input_channels,
                                     num_freqbins=num_freqbins,
                                     **model_config['network'])
        model_factory = vak.models.factory.ModelFactory(
            definition,
            vak.models.FrameClassificationModel,
        )
        model = model_factory.from_instances(network=network, labelmap=labelmap)
        model.to(device)

        eval_toml_path = specific_config_toml_path('eval', model_name, audio_format='cbin', annot_format='notmat')
        eval_cfg = vak.config.Config.from_toml_path(eval_toml_path)
        checkpoint_path = eval_cfg.eval.checkpoint_path

        # ---- actually test method
        sd_before = copy.deepcopy(model.state_dict())
        sd_before = {
            k: v.to(device) for k, v in sd_before.items()
        }
        ckpt = torch.load(checkpoint_path)
        sd_to_be_loaded = ckpt['state_dict']
        sd_to_be_loaded = {
            k: v.to(device) for k, v in sd_to_be_loaded.items()
        }

        model.load_state_dict_from_path(checkpoint_path)

        assert not all([
            torch.all(torch.eq(val, before_val))
            for val, before_val in zip(model.state_dict().values(), sd_before.values())]
        )
        assert all([
            torch.all(torch.eq(val, before_val))
            for val, before_val in zip(model.state_dict().values(), sd_to_be_loaded.values())]
        )
