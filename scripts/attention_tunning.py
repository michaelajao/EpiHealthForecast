
import sys
from pathlib import Path

sys.path.append('/mnt/c/Users/ajaoo/Desktop/Projects/hospitalization_research')
import time
import random
import shutil
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from sklearn.metrics import mean_absolute_error as mae
from src.dl.dataloaders import TimeSeriesDataModule
from src.dl.models import RNNConfig, Seq2SeqConfig, Seq2SeqModel, Seq2SeqwAttnConfig, Seq2SeqwAttnModel

HORIZON = 7
WINDOW = 14

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_data(source_data_path):
    source_data = Path(source_data_path)
    try:
        train_df = pd.read_csv(source_data / "data/raw/targetTransf_train.csv")
        test_df = pd.read_csv(source_data / "data/raw/targetTransf_test.csv")
        val_df = pd.read_csv(source_data / "data/raw/targetTransf_val.csv")
        return train_df, test_df, val_df
    except FileNotFoundError:
        print("File not found")
        exit(1)

def preprocess_data(train, val, test, target):
    train["type"] = "train"
    val["type"] = "val"
    test["type"] = "test"
    
    sample_df = pd.concat([
        train[[target, "type"]],
        val[[target, "type"]],
        test[[target, "type"]],
    ])
    sample_df[target] = sample_df[target].astype("float32")
    return sample_df

def objective(params, datamodule, test, target):
    attn, hidden_size, num_layers, learning_rate = params

    encoder_config = RNNConfig(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=True,
    ).__dict__
    decoder_input_size = 1 + hidden_size * 2

    decoder_config = RNNConfig(
        input_size=decoder_input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=True,
    ).__dict__

    rnn2rnn_config = Seq2SeqwAttnConfig(
        encoder_type="LSTM",
        decoder_type="LSTM",
        encoder_params=encoder_config,
        decoder_params=decoder_config,
        teacher_forcing_ratio=1,
        learning_rate=learning_rate,
    )

    model = Seq2SeqwAttnModel(attention_type=attn, config=rnn2rnn_config)
    trainer = pl.Trainer(
        min_epochs=5,
        max_epochs=100,
        devices=1, 
        accelerator="gpu",
        log_every_n_steps=10,
        callbacks=[pl.callbacks.EarlyStopping(monitor="valid_loss", patience=3)],
    )
    trainer.fit(model, datamodule)
    # Removing artifacts created during training
    shutil.rmtree("lightning_logs")
    pred = trainer.predict(model, datamodule.test_dataloader())
    pred = torch.cat(pred).squeeze().detach().numpy()
    pred = pred[0::HORIZON].ravel()
    pred = pred * datamodule.train.std + datamodule.train.mean
    test_truncated = test.iloc[-len(pred) :]
    if len(pred) != len(test_truncated):
        pred = pred[-len(test_truncated) :]
        
    actuals = test_truncated[target].values
    return mae(actuals, pred)

def neighbor(params, param_bounds):
    attn, hidden_size, num_layers, learning_rate = params
    
    attn = np.random.choice(param_bounds["attn"])
    hidden_size = np.random.randint(*param_bounds["hidden_size"])
    num_layers = np.random.randint(*param_bounds["num_layers"])
    learning_rate = np.random.uniform(*param_bounds["learning_rate"])
    learning_rate = round(learning_rate, 3)
    
    return [ attn, hidden_size, num_layers, learning_rate]


def cooling(temp, step):
    return temp * 0.99


def simulated_annealing(initial_params, initial_temp, steps, param_bounds, datamodule, test, target):
    current_params = initial_params
    current_cost = objective(current_params, datamodule, test, target)
    temp = initial_temp
    costs = [current_cost]
    temperatures = [initial_temp]

    for step in range(steps):
        new_params = neighbor(current_params, param_bounds)
        new_cost = objective(new_params, datamodule, test, target)

        if new_cost < current_cost or np.random.uniform(0, 1) < np.exp((current_cost - new_cost) / temp):
            current_params, current_cost = new_params, new_cost

        temp = cooling(temp, step)
        costs.append(current_cost)
        temperatures.append(temp)

    return current_params, costs, temperatures



def main():
    set_seeds()
    torch.set_float32_matmul_precision("medium")
    
    source_data = Path("/mnt/c/Users/ajaoo/Desktop/Projects/hospitalization_research")
    train_df, test_df, val_df = load_data(source_data)
    target = "covidOccupiedMVBeds"

    
    sample_df = preprocess_data(train_df, val_df, test_df, target)
    
    datamodule = TimeSeriesDataModule(
        data=sample_df[[target]],
        n_val=val_df.shape[0],
        n_test=test_df.shape[0],
        window=WINDOW,
        horizon=HORIZON,
        normalize="global",
        batch_size=32,
        num_workers=0,
    )
    datamodule.setup()
    
    param_bounds = {
        "attn": ["general", "concat", "additive", "dot"],
        "hidden_size": (32, 256),
        "num_layers": (1, 10),
        "learning_rate": (1e-6, 1e-2)
    }
    
    initial_params = ["general", 64, 1, 1e-2]
    initial_temp = 10000
    steps = 500
    
    start_time = time.time()
    best_params, costs, temperatures = simulated_annealing(
        initial_params, initial_temp, steps, param_bounds, datamodule, test_df, target
    )
    print("Best parameters found:", best_params)
    
    end_time = time.time()
    print(f"Total running time: {end_time - start_time} seconds")

    
    plt.plot(temperatures)
    plt.xlabel("Steps")
    plt.ylabel("Temperature")
    plt.title("Temperature Decay Plot")
    plt.savefig(source_data / "images/optimization/temperature_decay_plot4.png")
    plt.show()
    
    plt.plot(costs)
    plt.xlabel("Steps")
    plt.ylabel("Objective Value")
    plt.title("Convergence Plot")
    plt.savefig(source_data / "images/optimization/convergence_plot4.png")
    plt.show()

if __name__ == "__main__":
    main()