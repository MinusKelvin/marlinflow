from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess

from dataloader import BatchLoader, BucketingScheme
from model import Ice4Model
from time import time, strftime
from to_evalcpp import dump_result

import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WeightClipper:
    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "weight"):
            w = module.weight.data
            w = w.clamp(-1.98, 1.98)
            module.weight.data = w


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: BatchLoader,
    wdl: float,
    scale: float,
    epochs: int,
    training_id: str,
    lr_drop: list = [],
    lr_decay: float = 1.0,
    nndir: str = "",
) -> None:
    clipper = WeightClipper()
    running_loss = torch.zeros((1,), device=DEVICE)
    start_time = time()
    iterations = 0

    fens = 0
    epoch = 0
    last_loss = 0

    while epoch < epochs:
        new_epoch, batch = dataloader.read_batch(DEVICE)
        if new_epoch:
            epoch += 1
            last_loss = running_loss.item() / iterations
            optimizer.param_groups[0]["lr"] *= lr_decay
            if epoch in lr_drop:
                optimizer.param_groups[0]["lr"] *= 0.1
            print(f"epoch: {epoch}", end="\t")
            print(f"loss: ", end="")
            print(f"{last_loss:.4g}", end="\t")
            print(f"pos/s: {fens / (time() - start_time):.0f}", flush=True)

            running_loss = torch.zeros((1,), device=DEVICE)
            start_time = time()
            iterations = 0
            fens = 0

        expected = torch.sigmoid(batch.cp / scale) * (1 - wdl) + batch.wdl * wdl
        optimizer.zero_grad()
        prediction = model(batch)
        loss = torch.mean(torch.abs(prediction - expected) ** 2.6)
        loss.backward()

        with torch.no_grad():
            running_loss += loss
        optimizer.step()
        # model.apply(clipper)

        iterations += 1
        fens += batch.size

    train_result = {
        "loss": last_loss,
        "train_id": training_id,
        "params": {
            name: param.detach().cpu().numpy().tolist()
            for name, param in model.named_parameters()
        }
    }

    with open(f"{os.path.join(nndir, training_id)}.json", "w") as f:
        json.dump(train_result, f)

    dump_result(train_result)


def main():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--data", type=str, help="The data file"
    )
    parser.add_argument("--lr", type=float, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, help="Epochs to train for")
    parser.add_argument("--batch-size", type=int, default=16384, help="Batch size")
    parser.add_argument("--wdl", type=float, default=0.0, help="WDL weight to be used")
    parser.add_argument("--scale", type=float, help="WDL weight to be used")
    parser.add_argument("--nndir", type=str, default="", help="directory to store weights")
    parser.add_argument(
        "--lr-drop",
        type=int,
        action="append",
        help="The epoch learning rate will be dropped",
    )
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=1,
        help="Factor to multiply LR by every epoch"
    )
    parser.add_argument(
        "--convert",
        type=str,
        help="param map to convert"
    )
    parser.add_argument(
        "--train-id",
        type=str,
        help="training id"
    )
    args = parser.parse_args()

    if args.convert is not None:
        with open(args.convert) as f:
            result = json.load(f)
        dump_result(result)
        return

    assert args.scale is not None

    train_id = args.train_id or \
        strftime("%Y-%m-%d-%H-%M-%S-") + os.path.splitext(os.path.basename(args.data))[0]

    model = Ice4Model().to(DEVICE)

    dataloader = BatchLoader(
        lambda: [
#            subprocess.run(
#                ["./marlinflow-utils", "shuffle", "-i", args.data],
#                stdout=subprocess.DEVNULL
#            ),
            args.data
        ][-1],
        model.input_feature_set(),
        model.bucketing_scheme,
        args.batch_size
    )

    optimizer = torch.optim.Adam([
        param for param in model.parameters()
    ], lr=args.lr)

    print(f"train id: {train_id}")

    train(
        model,
        optimizer,
        dataloader,
        args.wdl,
        args.scale,
        args.epochs,
        train_id,
        lr_drop=args.lr_drop,
        lr_decay=args.lr_decay,
        nndir=args.nndir
    )


if __name__ == "__main__":
    main()
