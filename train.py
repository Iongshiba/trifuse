import os
import argparse
import torch
import wandb
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import MyDataSet
from main_model import HiFuse_Tiny
from trifuse import TriFuse_Tiny
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
        model = TriFuse_Tiny(num_classes=args.num_classes).to(device)
    elif args.model == "hifuse":
        model = HiFuse_Tiny(num_classes=args.num_classes).to(device)
    else:
        return

    if args.RESUME == False:
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(
                args.weights
            )
            weights_dict = torch.load(args.weights, map_location=device)["state_dict"]

            # Delete the weight of the relevant category
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            model.load_state_dict(weights_dict, strict=False)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # All weights except head are frozen
            if "head" not in name:
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
        path_checkpoint = "./model_weight/checkpoint/ckpt_best_100.pth"
        print("model continue train")
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        lr_scheduler.load_state_dict(checkpoint["lr_schedule"])

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

        if best_acc < val_acc:
            if not os.path.isdir("./model_weight"):
                os.mkdir("./model_weight")
            torch.save(model.state_dict(), "./model_weight/best_model.pth")
            print("Saved epoch{} as new best model".format(epoch))
            best_acc = val_acc

        if epoch % 10 == 0:
            print("epoch:", epoch)
            print("learning rate:", optimizer.state_dict()["param_groups"][0]["lr"])
            checkpoint = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "lr_schedule": lr_scheduler.state_dict(),
            }
            if not os.path.isdir("./model_weight/checkpoint"):
                os.mkdir("./model_weight/checkpoint")
            torch.save(
                checkpoint, "./model_weight/checkpoint/ckpt_best_%s.pth" % (str(epoch))
            )

        # add loss, acc and lr into tensorboard
        print("[epoch {}] accuracy: {}".format(epoch, round(val_acc, 3)))

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="trifuse")
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--RESUME", type=bool, default=False)

    parser.add_argument(
        "--train_data_path",
        type=str,
        default=r"C:\Users\trand\longg\document\selfstudy\hifuse\reference\HiFuse-main\kvasir",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default=r"C:\Users\trand\longg\document\selfstudy\hifuse\reference\HiFuse-main\kvasir",
    )

    parser.add_argument("--weights", type=str, default="", help="initial weights path")

    parser.add_argument("--freeze-layers", type=bool, default=False)
    parser.add_argument(
        "--device", default="cuda:0", help="device id (i.e. 0 or 0,1 or cpu)"
    )

    opt = parser.parse_args()

    main(opt)
