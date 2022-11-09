import os
import sys

import numpy as np
import pickle

from synthesis.model import dump_pairs

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from scripts.utils import load_config
from synthesis import NetworkBuilder
from synthesis.postprocess.align import align_fp


class Processor:
    def __init__(self, model_path, config_path, generator_type, discriminator_type, version, device='cpu'):
        super(Processor, self).__init__()
        self.device = device
        self.model = None
        self.model_version = version
        self._load_model(model_path, config_path, generator_type, discriminator_type)

    def _load_model(self, model_path, config_path, generator_type, discriminator_type):
        assert model_path is not None
        assert config_path is not None

        self.config = load_config(config_path)
        self.hidden_dim = self.config[generator_type]["latent_dimension"]
        self.batch_size = self.config['training']['batch_size']

        print(f'load checkpoint path:  {model_path}/{self.model_version}')
        checkpoint = torch.load(f"{model_path}/{self.model_version}")

        model = NetworkBuilder.build_network(
            self.config,
            network_type=generator_type,
            device=self.device,
            kernel_mask_dict=checkpoint.get('kernel_mask_dict')
        )

        discriminator = NetworkBuilder.build_network(
            self.config,
            network_type=discriminator_type,
            device=self.device
        )

        model.load_state_dict(checkpoint['vae_state_dict'])
        discriminator.load_state_dict(checkpoint['unet_state_dict'])

        print("Successfully Load Model...")

        total = sum([param.nelement() for param in model.parameters()]) + sum([param.nelement() for param in discriminator.parameters()])
        print("Number of parameter: % .2fM" % (total / 1e6))

        model.eval()

        self.model = model

    def forward(self):
        """
        predict furniture in room
        """
        with torch.no_grad():
            latent_code = torch.randn(self.batch_size, self.hidden_dim).to(self.device)
            boxes = self.model.sample(latent_code)

        return boxes

    @staticmethod
    def align(data):
        """
        align boxes with boundary
        """

        boxes_aligned, order = align_fp(
            data["boundary"],
            data["boxes"],
            data["type"],
            data["edges"],
            data["colors"],
            threshold=0.35
        )

        data["boxes_aligned"] = boxes_aligned
        data["order"] = order

        return data
