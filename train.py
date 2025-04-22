# filepath: c:\Users\trand\longg\document\selfstudy\hifuse\reference\HiFuse-main\train.py
import os
import argparse
import torch
import wandb
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import MyDataSet

# from main_model import HiFuse_Tiny # Assuming HiFuse also needs fpn_dim if used
from trifuse import TriFuse_Tiny  # Import updated TriFuse_Tiny
from utils import (
    read_train_data,
    read_val_data,
    create_lr_scheduler,
    get_params_groups,
    train_one_epoch,
    evaluate,
)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    wandb.login()
    logger = wandb.init(project="trifuse_kvasir", config=args)

    print(args)
    print(
        'Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/'
    )

    train_images_path, train_images_label = read_train_data(args.train_data_path)
    val_images_path, val_images_label = read_val_data(args.val_data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
    }

    train_dataset = MyDataSet(
        images_path=train_images_path,
        images_class=train_images_label,
        transform=data_transform["train"],
    )

    val_dataset = MyDataSet(
        images_path=val_images_path,
        images_class=val_images_label,
        transform=data_transform["val"],
    )

    batch_size = args.batch_size
    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, 8]
    )  # number of workers
    print("Using {} dataloader workers every process".format(nw))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn,
    )

    if args.model == "trifuse":
        # Pass fpn_dim from args to the model constructor
        model = TriFuse_Tiny(num_classes=args.num_classes, fpn_dim=args.fpn_dim).to(
            device
        )
    # elif args.model == "hifuse":
    # model = HiFuse_Tiny(num_classes=args.num_classes).to(device) # Potentially add fpn_dim here too if needed
    else:
        print(f"Model {args.model} not recognized.")
        return

    torch.save(model.state_dict, "trifuse_256.pth")

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))

    if args.RESUME == False:
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(
                args.weights
            )
            weights_dict = torch.load(args.weights, map_location=device)["state_dict"]

            # Delete the weight of the relevant category
            for k in list(weights_dict.keys()):
                # Adjust key check if head names changed due to fpn_dim
                if "head" in k or "linear" in k or "conv_head" in k or "conv_norm" in k:
                    del weights_dict[k]
            print("Loading pre-trained weights, ignoring head layers.")
            model.load_state_dict(weights_dict, strict=False)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # Adjust freeze logic if head names changed
            if not (
                "head" in name
                or "linear" in name
                or "conv_head" in name
                or "conv_norm" in name
            ):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(
        optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=1
    )

    best_acc = 0.0
    start_epoch = 0

    if args.RESUME:
        # Ensure checkpoint path exists and is correct
        path_checkpoint = "./model_weight/checkpoint/ckpt_best_100.pth"  # Example path, adjust if needed
        if os.path.exists(path_checkpoint):
            print("Resuming training from checkpoint:", path_checkpoint)
            checkpoint = torch.load(path_checkpoint, map_location=device)
            # Load model state dict carefully, especially if architecture changed
            model.load_state_dict(checkpoint["net"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            lr_scheduler.load_state_dict(checkpoint["lr_schedule"])
            # Load best_acc if saved in checkpoint
            if "best_acc" in checkpoint:
                best_acc = checkpoint["best_acc"]
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found at {path_checkpoint}, starting from scratch.")

    for epoch in range(start_epoch + 1, args.epochs + 1):

        # train
        train_stats = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            lr_scheduler=lr_scheduler,
        )

        logger.log(train_stats)

        # validate
        val_stats = evaluate(
            model=model, data_loader=val_loader, device=device, epoch=epoch
        )

        logger.log(val_stats)
        val_acc = val_stats["eval/accuracy"]

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        # Save checkpoint logic
        save_dict = {
            "epoch": epoch,
            "net": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_schedule": lr_scheduler.state_dict(),
            "best_acc": best_acc,
            "args": args,
        }

        if not os.path.isdir("./model_weight"):
            os.makedirs("./model_weight")
        if not os.path.isdir("./model_weight/checkpoint"):
            os.makedirs("./model_weight/checkpoint")

        if is_best:
            torch.save(save_dict, "./model_weight/best_model.pth")
            print(
                f"Saved epoch {epoch} as new best model with accuracy: {best_acc:.4f}"
            )

        if (
            epoch % 10 == 0 or epoch == args.epochs
        ):  # Save checkpoint every 10 epochs and at the end
            print("epoch:", epoch)
            print("learning rate:", optimizer.state_dict()["param_groups"][0]["lr"])
            torch.save(save_dict, f"./model_weight/checkpoint/ckpt_epoch_{epoch}.pth")
            print(f"Saved checkpoint for epoch {epoch}")

        # add loss, acc and lr into tensorboard
        print("[epoch {}] accuracy: {}".format(epoch, round(val_acc, 3)))
    logger.finish()  # Finish wandb run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="trifuse",
        help="Model name (e.g., trifuse, hifuse)",
    )
    parser.add_argument(
        "--num_classes", type=int, default=8, help="Number of output classes"
    )
    parser.add_argument(
        "--fpn-dim", type=int, default=256, help="Dimension of FPN output channels"
    )  # Added fpn_dim arg
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    parser.add_argument(
        "--RESUME", action="store_true", help="Resume training from checkpoint"
    )  # Use action='store_true'

    # Default paths updated for clarity, ensure they exist or adjust as needed
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=r"C:\Users\trand\longg\document\selfstudy\hifuse\reference\HiFuse-main\kvasir\train",  # Example path
        help="Path to the training dataset root directory",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default=r"C:\Users\trand\longg\document\selfstudy\hifuse\reference\HiFuse-main\kvasir\val",  # Example path
        help="Path to the validation dataset root directory",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Initial weights path (for transfer learning, ignored if RESUME is True)",
    )

    parser.add_argument(
        "--freeze-layers", action="store_true", help="Freeze layers except the head"
    )  # Use action='store_true'
    parser.add_argument(
        "--device", default="cuda:0", help="Device id (e.g., cuda:0 or cpu)"
    )

    opt = parser.parse_args()

    # Ensure data paths exist before starting
    assert os.path.exists(
        opt.train_data_path
    ), f"Training data path not found: {opt.train_data_path}"
    assert os.path.exists(
        opt.val_data_path
    ), f"Validation data path not found: {opt.val_data_path}"

    main(opt)
