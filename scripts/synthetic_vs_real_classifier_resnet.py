import argparse
import os
import sys

from PIL import Image

import numpy as np
import torch
import torch.utils.data.dataset
from torchvision import models
from torchvision.transforms import transforms

train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomAffine(5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class SyntheticVRealDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, real, fake, tag):
        self.tag = tag
        self.real = real
        self.n1 = len(real)
        self.fake = fake
        self.n2 = len(fake)

        self.transform = train_transform

    def __len__(self):
        return len(self.real) + len(self.fake)

    def __getitem__(self, idx):
        if self.tag == 'train':
            if idx < self.n1:
                image_path = self.real[idx]
                label = 1
            elif idx < self.n1 + self.n2:
                image_path = self.fake[idx - self.n1][0]
                label = 0
        else:
            if idx < self.n1:
                image_path = self.real[idx]
                label = 1
            # elif idx < self.n1 + self.n2:
            #     image_path = self.fake[idx - self.n1][1]
            #     label = 0
            elif idx < self.n1 + int(self.n2 / 2):
                image_path = self.fake[idx - self.n1][0]
                label = 0
            else:
                image_path = self.fake[idx - self.n1 - int(self.n2 / 2)][1]
                label = 0
            # if idx < self.n1:
            #     image_path = self.fake[idx][0]
            #     label = 0
            # elif idx < self.n1 + self.n2:
            #     image_path = self.fake[idx - self.n1][1]
            #     label = 0

        # if idx < self.n1:
        #     image_path = self.real[idx]
        #     label = 1
        # # elif idx < self.n1 + int(self.n2 / 2):
        # elif idx < self.n1 + self.n2:
        #     image_path = self.fake[idx - self.n1][0]
        #     label = 0
        # else:
        #     image_path = self.fake[idx - self.n1 - int(self.n2 / 2)][1]
        #     label = 0

        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        return img, torch.tensor([label], dtype=torch.float)

class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(pretrained=True)
        self.fc = torch.nn.Linear(1000, 1)

    def forward(self, x):
        x = self.model.forward(x)
        x = self.fc(x.view(len(x), -1))
        x = torch.sigmoid(x)
        return x


class AverageMeter:
    def __init__(self):
        self._value = 0
        self._cnt = 0

    def __iadd__(self, x):
        if torch.is_tensor(x):
            self._value += x.sum().item()
            self._cnt += x.numel()
        else:
            self._value += x
            self._cnt += 1
        return self

    @property
    def value(self):
        return self._value / self._cnt


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Train a classifier to discriminate between real "
                     "and synthetic rooms")
    )

    parser.add_argument(
        "--path_to_renderings",
        default="/data/render_scene/scene-synth",
        help="Path to the folder containing the synthesized"
    )
    parser.add_argument(
        "--dataset_type",
        default="bedroom",
        choices=[
            "bedroom",
            "livingroom",
            "diningroom",
            "library"
        ],
        help="The type of dataset filtering to be used"
    )

    parser.add_argument(
        "--tag",
        default="baseline",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Set the batch size for training and evaluating (default: 256)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Set the PyTorch data loader workers (default: 0)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Train for that many epochs (default: 10)"
    )
    parser.add_argument(
        "--output_directory",
        default="/data/fake_real_classifier",
        help="Path to the output directory"
    )
    args = parser.parse_args(argv)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create Real datasets
    dataset_images = [
        f"{args.path_to_renderings}/raw_{args.dataset_type}/furniture_only/{f}"
        for f in os.listdir(f"{args.path_to_renderings}/raw_{args.dataset_type}/furniture_only")
        if any(f.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
    ]

    gen_images = [
        f"{args.path_to_renderings}/{args.dataset_type}/{args.tag}/furniture_only/{f}"
        for f in os.listdir(f"{args.path_to_renderings}/{args.dataset_type}/{args.tag}/furniture_only")
        if any(f.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
    ]

    processed_images = [
        f"{args.path_to_renderings}/{args.dataset_type}/{args.tag}/furniture_complete/{f}"
        for f in os.listdir(f"{args.path_to_renderings}/{args.dataset_type}/{args.tag}/furniture_complete")
        if any(f.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
    ]
    # np.random.shuffle(synthesized_images)
    fake_images = list(zip(gen_images, processed_images))

    np.random.shuffle(dataset_images)
    np.random.shuffle(fake_images)

    train_dataset = SyntheticVRealDataset(dataset_images[0:-500], fake_images[0:-500], 'train')
    test_dataset = SyntheticVRealDataset(dataset_images[-500:-1], fake_images[-500:-1], 'test')


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Create the model
    model = ResNet()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    acc_scores = []
    for _ in range(10):
        for e in range(args.epochs):
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            correct = 0.
            total = 0.
            for i, (x, y) in enumerate(train_dataloader):
                model.train()
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_hat = model(x)
                loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
                acc = (torch.abs(y - y_hat) < 0.5).float()
                loss.backward()
                optimizer.step()

                loss_meter += loss
                acc_meter += acc.mean()

                for label_idx in range(len(y)):
                    if y[label_idx] == 1:
                        correct += acc[label_idx].item()
                        total += 1

                msg = "{: 3d} loss: {:.4f} - acc: {:.4f} - recall: {:.4f}".format(
                    i, loss_meter.value, acc_meter.value, correct / total
                )
                print(msg + "\b" * len(msg), end="", flush=True)
            print()

            if (e + 1) % 5 == 0:
                with torch.no_grad():
                    model.eval()
                    loss_meter = AverageMeter()
                    acc_meter = AverageMeter()
                    correct = 0.
                    total = 0.
                    for i, (x, y) in enumerate(test_dataloader):
                        x = x.to(device)
                        y = y.to(device)
                        y_hat = model(x)

                        loss = torch.nn.functional.binary_cross_entropy(
                            y_hat, y
                        )
                        acc = (torch.abs(y - y_hat) < 0.5).float()
                        # acc = (y_hat.ge(0.5) == y).float()

                        loss_meter += loss
                        acc_meter += acc.mean()

                        for label_idx in range(len(y)):
                            if y[label_idx] == 1:
                                correct += acc[label_idx].item()
                                total += 1

                        msg = "{: 3d} val_loss: {:.4f} - val_acc: {:.4f} - val_recall: {:.4f}".format(
                            i, loss_meter.value, acc_meter.value, correct / total if total != 0. else 0.
                        )
                        print(msg + "\b" * len(msg), end="", flush=True)
                    print()
        acc_scores.append(acc_meter.value)

    print("acc_mean: " + str(sum(acc_scores) / len(acc_scores)))
    print("acc_std: " + str(np.std(acc_scores)))


if __name__ == "__main__":
    main(None)
