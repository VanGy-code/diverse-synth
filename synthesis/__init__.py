import torch
from synthesis.model.UNet3p import UNet3Plus
from synthesis.model.EnhancedBetaTCVAE import EnhancedBetaTCVAE
try:
    from radam import RAdam
except ImportError:
    pass


class NetworkBuilder:
    @staticmethod
    def train_on_batch(model, optimizer, sample_params, config):
        # Make sure that everything has the correct size
        optimizer.zero_grad()
        x_pred = model(sample_params)
        # Compute the loss
        loss = x_pred.reconstruction_loss(sample_params, sample_params["lengths"])
        # Do the backpropagation
        loss.backward()
        # Do the update
        optimizer.step()

        return loss.item()

    @staticmethod
    @torch.no_grad()
    def validate_on_batch(model, sample_params, config):
        x_pred = model(sample_params)
        # Compute the loss
        loss = x_pred.reconstruction_loss(sample_params, sample_params["lengths"])
        return loss.item()

    @staticmethod
    def build_network(
            config,
            weight_file=None,
            device="cpu",
            network_type=None,
            **kwargs
    ):
        if network_type == "EnhancedBetaTCVAE":
            network = EnhancedBetaTCVAE(
                data_config=config["data"],
                model_config=config["EnhancedBetaTCVAE"],
                kernel_mask_dict=kwargs['kernel_mask_dict']
            )
        elif network_type == 'UNet3P':
            network = UNet3Plus(
                data_config=config["data"],
                model_config=config["UNet3Plus"],
            )
        else:
            raise NotImplementedError()

        if weight_file is not None:
            print("Loading weight file from {}".format(weight_file))
            network.load_state_dict(
                torch.load(weight_file, map_location=device)
            )
        network.to(device)
        return network


class Optimizer:
    @staticmethod
    def build_optimizer(parameters, lr=1e-3, optimizer='Adam', momentum=0, weight_decay=0, betas=None):

        if optimizer == "SGD":
            return torch.optim.SGD(
                parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        elif optimizer == "Adam":
            if betas is not None:
                return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, betas=betas)
            else:
                return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer == "RAdam":
            if betas is not None:
                return RAdam(parameters, lr=lr, weight_decay=weight_decay, betas=betas)
            else:
                return RAdam(parameters, lr=lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError()
