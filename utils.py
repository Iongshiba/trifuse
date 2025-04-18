import os
import sys
import json
import shutil
import pickle
import random
import math
from PIL import Image
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


def split_dataset(root_dir, train_dir="train", val_dir="val", split_ratio=0.5, seed=42):
    """
    Split dataset into training and validation sets.

    Args:
        root_dir: Path to the root directory containing class folders
        train_dir: Path where training data will be copied
        val_dir: Path where validation data will be copied
        split_ratio: Ratio of data for training (0.5 means 50% train, 50% val)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_folder in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_folder)

        if not os.path.isdir(class_path):
            continue

        train_class_path = os.path.join(train_dir, class_folder)
        val_class_path = os.path.join(val_dir, class_folder)

        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(val_class_path, exist_ok=True)

        # Get all image files
        images = [
            f
            for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f))
        ]

        random.shuffle(images)

        split_point = int(len(images) * split_ratio)

        train_images = images[:split_point]
        val_images = images[split_point:]

        for img in train_images:
            shutil.copy(
                os.path.join(class_path, img), os.path.join(train_class_path, img)
            )

        for img in val_images:
            shutil.copy(
                os.path.join(class_path, img), os.path.join(val_class_path, img)
            )

    print(f"Dataset split complete: {train_dir} and {val_dir}")


def read_train_data(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    category = [
        cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root, cls))
    ]
    category.sort()
    class_indices = dict((k, v) for v, k in enumerate(category))
    json_str = json.dumps(
        dict((val, key) for key, val in class_indices.items()), indent=4
    )

    with open("class_indices.json", "w") as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []

    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cls in category:
        cls_path = os.path.join(root, cls)
        images = [
            os.path.join(root, cls, i)
            for i in os.listdir(cls_path)
            if os.path.splitext(i)[-1] in supported
        ]

        image_class = class_indices[cls]

        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(image_class)

    print("{} images for training.".format(len(train_images_path)))

    return train_images_path, train_images_label


def read_val_data(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    category = [
        cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root, cls))
    ]
    category.sort()
    class_indices = dict((k, v) for v, k in enumerate(category))

    val_images_path = []
    val_images_label = []

    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cls in category:
        cls_path = os.path.join(root, cls)
        images = [
            os.path.join(root, cls, i)
            for i in os.listdir(cls_path)
            if os.path.splitext(i)[-1] in supported
        ]
        image_class = class_indices[cls]

        for img_path in images:
            val_images_path.append(img_path)
            val_images_label.append(image_class)

    print("{} images for validation.".format(len(val_images_path)))

    return val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = "./class_indices.json"
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, "r")
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype("uint8"))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, "wb") as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, "rb") as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = (
            "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_num.item() / sample_num,
                optimizer.param_groups[0]["lr"],
            )
        )

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training ", loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    stats = {
        "train/loss": accu_loss.item() / (step + 1),
        "train/accuracy": accu_num.item() / sample_num,
        "train/learning rate": optimizer.param_groups[0]["lr"],
    }

    return stats


class MyDataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    all_preds = []
    all_labels = []

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.numpy())

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num
        )

    precision, recall, f1 = compute_metrics(all_preds, all_labels)

    stats = {
        "eval/loss": accu_loss.item() / (step + 1),
        "eval/accuracy": accu_num.item() / sample_num,
        "eval/precision": precision,
        "eval/recall": recall,
        "eval/f1": f1,
    }

    return stats


def compute_metrics(preds, labels):
    unique_labels = set(labels)

    precision_sum = 0
    recall_sum = 0
    f1_sum = 0

    for cls in unique_labels:
        tp = sum([1 for p, l in zip(preds, labels) if p == cls and l == cls])
        fp = sum([1 for p, l in zip(preds, labels) if p == cls and l != cls])
        fn = sum([1 for p, l in zip(preds, labels) if p != cls and l == cls])

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    num_classes = len(unique_labels)
    avg_precision = precision_sum / num_classes
    avg_recall = recall_sum / num_classes
    avg_f1 = f1_sum / num_classes

    return avg_precision, avg_recall, avg_f1


def create_lr_scheduler(
    optimizer,
    num_step: int,
    epochs: int,
    warmup=True,
    warmup_epochs=1,
    warmup_factor=1e-3,
    end_factor=1e-2,
):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = x - warmup_epochs * num_step
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (
                1 - end_factor
            ) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    parameter_group_vars = {
        "decay": {"params": [], "weight_decay": weight_decay},
        "no_decay": {"params": [], "weight_decay": 0.0},
    }

    parameter_group_names = {
        "decay": {"params": [], "weight_decay": weight_decay},
        "no_decay": {"params": [], "weight_decay": 0.0},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


if __name__ == "__main__":
    split_dataset(
        r"C:\Users\trand\longg\document\selfstudy\hifuse\reference\HiFuse-main\kvasir",
        r"C:\Users\trand\longg\document\selfstudy\hifuse\reference\HiFuse-main\kvasir\train",
        r"C:\Users\trand\longg\document\selfstudy\hifuse\reference\HiFuse-main\kvasir\val",
    )
