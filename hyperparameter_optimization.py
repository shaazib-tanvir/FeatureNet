from dataloader import IUXRayDataset, num_cls, conditions_table, conditions
from torchvision.transforms.v2 import Resize, CenterCrop, ToImage, ToDtype, Compose
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from models.featurenet import FeatureNet
from tqdm import tqdm
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import pickle
import os.path

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocabulary-path", type=str, help="the path of the vocabulary file generated by preprocessing/build_vocab.py", default="data/vocab.pkl")
    parser.add_argument("--image-dir", type=str, help="the path of the images directory", default="data/images/images_normalized")
    parser.add_argument("--train-data-csv", type=str, help="the csv file generated by preprocessing/iu_preprocessing.py containing training data annotations", default="data/iu_train_data.csv")
    parser.add_argument("--val-data-csv", type=str, help="the csv file generated by preprocessing/iu_preprocessing.py containing valdation data annotaitons", default="data/iu_val_data.csv")
    parser.add_argument("--batch-size", type=int, help="the batch size of the data loader", default=8)
    parser.add_argument("--patch-size", type=int, help="the patch size used by bagnet", default=33)
    parser.add_argument("--epochs", type=int, help="the number of epochs to train the model for", default=20)
    return parser.parse_args()


def train(args, model, dataloader, optimizer, lossfn):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch, (image, label, reports) in progress_bar:
        image = image.to("cuda")
        label = label.to("cuda")

        optimizer.zero_grad()
        prediction = model(image)
        loss = lossfn(prediction, label)
        loss.backward()
        optimizer.step()

        running_loss += loss

        progress_bar.set_description(f"Loss: {running_loss / (batch + 1)}")

    return running_loss / len(dataloader)


def validatate(args, model, dataloader, lossfn):
    model.eval()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for batch, (image, label, reports) in progress_bar:
            image = image.to("cuda")
            label = label.to("cuda")

            prediction = model(image)
            loss = lossfn(prediction, label)

            running_loss += loss

            progress_bar.set_description(f"Loss: {running_loss / (batch + 1)}")

    return running_loss / len(dataloader)


def objective(config):
    args = parse()
    
    vocabulary = None
    with open(args.vocabulary_path, mode="rb") as file:
        vocabulary = pickle.load(file)

    train_dataset = IUXRayDataset(image_dir=args.image_dir, image_list_file=args.train_data_csv,
                                  transform=Compose([Resize(224), CenterCrop(224), ToImage(), ToDtype(torch.float32, scale=True)]))
    val_dataset = IUXRayDataset(image_dir=args.image_dir, image_list_file=args.val_data_csv,
                                  transform=Compose([Resize(224), CenterCrop(224), ToImage(), ToDtype(torch.float32, scale=True)]))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    model = FeatureNet(num_cls, args.patch_size).to("cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

    lossfn = torch.nn.CrossEntropyLoss()

    train_losses = np.zeros(args.epochs, dtype=np.float32)
    validation_losses = np.zeros(args.epochs, dtype=np.float32)
    for epoch in range(args.epochs):
        print(f"Epoch \033[32m{epoch}\033[0m / \033[32m{args.epochs - 1}\033[0m\n")
        train_losses[epoch] = train(args, model, train_dataloader, optimizer, lossfn)
        validation_losses[epoch] = validatate(args, model, val_dataloader, lossfn)

    train.report({"loss": min(validation_losses)})


if __name__ == "__main__":
    search_space = {
        "lr": tune.sample_from(lambda spec: 10 ** (-5 * np.random.rand())),
        "momentum": tune.uniform(0.1, 0.9),
        "weight_decay": tune.sample_from(lambda spec: 10 ** (-3 - 2 * np.random.rand())),
        "gamma": tune.uniform(0.1, 0.9),
        "step_size": tune.choice([1, 5, 10])
    }

    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            num_samples=20,
            scheduler=ASHAScheduler(metric="loss", mode="min"),
        ),
        param_space=search_space
    )

    results = tuner.fit()
    print(results.get_best_result(metric="loss", mode="min").config)
    #objective({"lr": 1e-2, "momentum": 0.0, "weight_decay": 0.0, "gamma": 0.1, "step_size": 1})
