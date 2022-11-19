import argparse
import logging
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import load_config
from synthesis.datasets.Common import filter_function
from synthesis.datasets.FrontDataset import get_encoded_dataset
from synthesis import NetworkBuilder, Optimizer


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a generative model on bounding boxes"
    )

    parser.add_argument(
        "--config_file",
        default="../config/bedroom_config.yaml",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )

    parser.add_argument('--lr_decay_steps', default='1000,2000',
                        help='When to decay the learning rate (in epochs) [default: 80,120,160]')

    parser.add_argument('--lr_decay_rates', default='0.1,0.1',
                        help='Decay rates for lr decay [default: 0.1,0.1,0.1]')

    parser.add_argument(
        "--n_processes",
        type=int,
        default=8,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
        help="Seed for the PRNG"
    )

    parser.add_argument(
        "--generator_type",
        default='EnhancedVAE',
    )

    parser.add_argument(
        "--discriminator_type",
        default='UNet3P',
    )
    
    parser.add_argument(
        "--wandb_entity",
        help="wandb username"
    )
    

    args = parser.parse_args(argv)

    # Disable trimesh logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Parse the config file
    config = load_config(args.config_file)

    # ================= Step0 : Init Wandb  =======================
    wandb.config = {
        "learning_rate": config['training'].get('tag'),
        "epochs": config['training'].get('epochs'),
        "batch_size": config['training'].get('batch_size'),
        'lr_decay_steps': [int(x) for x in args.lr_decay_steps.split(',')],
        'lr_decay_rates': [float(x) for x in args.lr_decay_rates.split(',')]
    }
    wandb.init(
        project="diverse-synth",
        entity=args.wandb_entity,
        name=config['training'].get('tag'),
        config=wandb.config,
    )

    # ================= Step1 : preparation =======================

    LR_DECAY_STEPS = [int(x) for x in args.lr_decay_steps.split(',')]
    LR_DECAY_RATES = [float(x) for x in args.lr_decay_rates.split(',')]

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)
    wandb.config.update({"device": device})

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(config['training']['checkpoint_dir']):
        os.makedirs(config['training']['checkpoint_dir'])

    # Create an experiment directory using the experiment_tag
    experiment_tag = config['training'].get('tag')
    experiment_directory = f"{config['training']['checkpoint_dir']}/{experiment_tag}"
    checkpoint_path = f"{experiment_directory}/checkpoint.tar"

    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Create dataset
    dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        path_to_bounds=None,
        split=config["training"].get("splits", ["train", "val"])
    )
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 16),
        num_workers=args.n_processes,
        worker_init_fn=dataset.worker_init_fn,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    
    print("Loaded {} training scenes".format(
        len(train_dataset))
    )
    
    # print(train_dataset.num_class)
    # print("Loaded {} training scenes with {} object types".format(
    #     len(train_dataset), train_dataset.num_class)
    # )
    # print("Training set has {} bounds".format(train_dataset.bounds))

    # path_to_bounds = os.path.join(experiment_directory, "bounds.npz")
    # np.savez(
    #     path_to_bounds,
    #     sizes=train_dataset.bounds["sizes"],
    #     translations=train_dataset.bounds["translations"],
    #     angles=train_dataset.bounds["angles"]
    # )
    # print("Saved the dataset bounds in {}".format(path_to_bounds))

    # validation_dataset = get_encoded_dataset(
    #     config["data"],
    #     filter_function(
    #         config["data"],
    #         split=config["validation"].get("splits", ["test"])
    #     ),
    #     path_to_bounds=path_to_bounds,
    #     split=config["validation"].get("splits", ["test"])
    # )

    val_loader = DataLoader(
        # validation_dataset,
        test_dataset,
        batch_size=config["validation"].get("batch_size", 16),
        num_workers=args.n_processes,
        worker_init_fn=dataset.worker_init_fn,
        # worker_init_fn=validation_dataset.worker_init_fn,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )
    
    
    print("Loaded {} validation scenes".format(
        len(test_dataset))
    )

    # print("Loaded {} validation scenes with {} object types".format(
    #     len(validation_dataset), validation_dataset.num_class)
    # )
    # print("Validation set has {} bounds".format(validation_dataset.bounds))

    # assert train_dataset.class_labels == validation_dataset.class_labels

    # ================= Step2 : Build Network =======================
    # Build the network architecture to be used for training

    kernel_mask_dict = None
    checkpoint = None
    # load savepoint
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        print('load checkpoint path: %s' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        kernel_mask_dict = checkpoint.get('kernel_mask_dict')


    vae = NetworkBuilder.build_network(
        config,
        network_type=args.generator_type,
        device=device,
        kernel_mask_dict=kernel_mask_dict)

    unet = NetworkBuilder.build_network(config, network_type=args.discriminator_type, device=device)


    vae_optimizer = Optimizer.build_optimizer(vae.parameters(),
                                              lr=config['training'].get("lr", 1e-3),
                                              optimizer=config['training'].get("optimizer", "Adam"),
                                              momentum=config['training'].get("momentum", 0.9),
                                              weight_decay=config['training'].get("weight_decay", 0.0),
                                              betas=None)

    unet_optimizer = Optimizer.build_optimizer(unet.parameters(),
                                               lr=config['training'].get("lr", 1e-3),
                                               optimizer=config['training'].get("optimizer", "Adam"),
                                               momentum=config['training'].get("momentum", 0.9),
                                               weight_decay=config['training'].get("weight_decay", 0.0),
                                               betas=None)

    vae_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(vae_optimizer, milestones=LR_DECAY_STEPS,
                                                                   gamma=0.1)

    unet_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(unet_optimizer, milestones=LR_DECAY_STEPS,
                                                                    gamma=0.1)

    start_epoch = 0
    epochs = config["training"].get("epochs", 100)
    # save checkpoint every 10 epochs
    save_every = config["training"].get("save_frequency", 10)

    # load savepoint
    # if checkpoint_path is not None and os.path.isfile(checkpoint_path):
    if checkpoint is not None:
        vae.load_state_dict(checkpoint['vae_state_dict'])
        unet.load_state_dict(checkpoint['unet_state_dict'])
        vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
        unet_optimizer.load_state_dict(checkpoint['unet_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print("Successfully Load Model...")

    # ================= Step3 : Training Model =======================
    for epoch in range(start_epoch, epochs + 1):
        print('**** EPOCH %03d ****' % epoch)
        lr = vae_optimizer_scheduler.get_last_lr()[0]
        print('Current learning rate: %f' % lr)

        np.random.seed(0)

        # ======================= Train parse ========================
        stat_dict = {}
        # loss_dict = {}
        vae.train()
        unet.train()

        for batch_idx, batch_data_label in enumerate(tqdm(train_loader)):
            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].to(device)

            # Forward pass
            vae_optimizer.zero_grad()
            unet.zero_grad()

            inputs_abs = batch_data_label['x_abs']

            labels_abs = batch_data_label['x_abs']
            labels_rel = batch_data_label['x_rel']

            package = vae(inputs_abs)
            feature_x = unet(package[-1].detach())

            vae_loss_dict, current_idx, ground_truth_idx, reconstruct_idx = vae.loss_function(
                ground_truth=labels_abs,
                package=package,
                dataset_size=len(train_dataset),
            )

            unet_loss_dict = unet.loss_function(
                room_type=config['data']['room_type'],
                ground_truth=labels_rel,
                reconstruct_x=feature_x,
                batch_idx=current_idx,
                reconstruct_idx=reconstruct_idx,
                ground_truth_idx=ground_truth_idx
            )

            loss_dict = {**vae_loss_dict, **unet_loss_dict}
            total_loss = torch.zeros(1).to(device=labels_abs.device)
            for key, value in loss_dict.items():
                if key == 'mi_loss': continue
                total_loss += value
            loss_dict['total_loss'] = total_loss

            total_loss.backward()
            vae_optimizer.step()
            unet_optimizer.step()

            # Accumulate statistics and print out
            for key in loss_dict:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += loss_dict[key].item()

        print('epoch: %03d:' % epoch, end=' ')

        train_log = {
            key: stat_dict[key] / float(len(train_loader) + 1) for key in sorted(stat_dict.keys())
        }
        train_log.update({"epoch": epoch, "current_lr": lr})
        wandb.log(train_log)
        for key in sorted(stat_dict.keys()):
            print('%s: %f |' % (key, stat_dict[key] / float(len(train_loader) + 1)), end=' ')
            stat_dict[key] = 0
        print()
        # ======================= eval parse ========================
        save_dict = {
            # after training one epoch, the start_epoch should be epoch+1
            'epoch': epoch + 1,
            'vae_optimizer_state_dict': vae_optimizer.state_dict(),
            'unet_optimizer_state_dict': unet_optimizer.state_dict(),
        }
        print('================ In evaluation mode ================')

        eval_stat_dict = {}

        vae.eval()
        unet.eval()

        for batch_idx, batch_data_label in enumerate(tqdm(val_loader)):
            for k in batch_data_label:
                batch_data_label[k] = batch_data_label[k].to(device)

            # Forward pass
            inputs_abs = batch_data_label['x_abs']
            labels_abs = batch_data_label['x_abs']
            labels_rel = batch_data_label['x_rel']

            with torch.no_grad():
                package = vae(inputs_abs)
                feature_x = unet(package[-1].detach())

                vae_loss_dict, current_idx, ground_truth_idx, reconstruct_idx = vae.loss_function(
                    ground_truth=labels_abs,
                    package=package,
                    dataset_size=len(train_dataset),
                )

                unet_loss_dict = unet.loss_function(
                    room_type=config['data']['room_type'],
                    ground_truth=labels_rel,
                    reconstruct_x=feature_x,
                    batch_idx=current_idx,
                    reconstruct_idx=reconstruct_idx,
                    ground_truth_idx=ground_truth_idx
                )
            # Accumulate statistics and print out
            eval_loss_dict = {**vae_loss_dict, **unet_loss_dict}
            total_loss = torch.zeros(1).to(device)
            for key, value in eval_loss_dict.items():
                if key == 'mi_loss': continue
                total_loss += value
            eval_loss_dict['total_loss'] = total_loss

            # Accumulate statistics and print out
            for key in eval_loss_dict:
                if key not in eval_stat_dict:
                    eval_stat_dict['eval_' + key] = 0
                eval_stat_dict['eval_' + key] += eval_loss_dict[key].item()

        eval_log = {
            key: eval_stat_dict[key] / float(len(val_loader) + 1) for key in sorted(eval_stat_dict.keys())
        }
        mean_loss = eval_stat_dict['eval_total_loss'] / float(len(val_loader) + 1)
        eval_log.update({"epoch": epoch, "mean_loss": mean_loss})
        wandb.log(eval_log)
        for key in sorted(eval_stat_dict.keys()):
            print(
                'eval %s: %f | ' % (key, eval_stat_dict[key] / (len(val_loader) + 1))
                , end=' ')
            eval_stat_dict[key] = 0
        print()

        # Save checkpoint

        # with nn.DataParallel() the net is added as a submodule of DataParallel
        try:
            save_dict['vae_state_dict'] = vae.module.state_dict()
        except Exception:
            save_dict['vae_state_dict'] = vae.state_dict()

        try:
            save_dict['unet_state_dict'] = unet.module.state_dict()
        except Exception:
            save_dict['unet_state_dict'] = unet.state_dict()

        if epoch % save_every == 0:
            if args.generator_type not in ['VAE']:
                save_dict['kernel_mask_dict'] = vae.kernel_mask_dict
            torch.save(save_dict,
                        f"{config['training']['checkpoint_dir']}/{config['training']['tag']}/checkpoint_eval{epoch}.tar")

        vae_optimizer_scheduler.step()
        unet_optimizer_scheduler.step()

        save_dict = {
            'vae_optimizer_state_dict': vae_optimizer.state_dict(),
            'unet_optimizer_state_dict': unet_optimizer.state_dict(),
        }

        if args.generator_type not in ['VAE']:
            save_dict['kernel_mask_dict'] = vae.kernel_mask_dict
        torch.save(save_dict, f"{config['training']['checkpoint_dir']}/{config['training']['tag']}/final.tar")
    wandb.finish()


if __name__ == '__main__':
    main(sys.argv[1:])
