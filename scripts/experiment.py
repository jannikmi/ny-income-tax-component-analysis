# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Income Tax Analysis NY
#
# Global Definitions

# %%
# %matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
import math
import pandas as pd
import os
from pathlib import Path
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(os.path.abspath('')).parent
DATA_FOLDER = PROJECT_ROOT / "data"
REPORTS_FOLDER = PROJECT_ROOT / "report"
CSV_FILE = DATA_FOLDER / "personal-income-tax-filers-summary-dataset-2-major-items-and-income-deduction-components-by-place-of-1.csv"

COL_TARGET = "Taxes Paid"
COL_TAX_YEAR = "Tax Year"
COL_CARDINALITY = "Number of Returns"
COL_COUNTRY = "Country"
COL_RESIDENCE = "Place of Residence"
COL_INCOME_RANGE = "NY Adjusted Gross Income Range (Fed.Col)"
COL_SOURCE = "source"
COLS_CATEGORICAL = ["NYS Residency Status", COL_RESIDENCE, "County", "State", COL_COUNTRY,
                    COL_INCOME_RANGE]

COLS_MONETARY = [
    # types of income:
    # 'Taxable Income',
    'Wage and Salary Income',
    'Gain from Capital&Supplemental Income',
    # 'Loss from Capital&Supplemental Income',
    'Gain from Rent,Royalties,Prtnrshp,Estates,Trusts Income',
    # 'Loss from Rent,Royalties,Prtnrshp,Estates,Trusts Income',
    'Gain from Business & Farm Income',
    # 'Loss from Business & Farm Income',
    # 'Pension/Annuity & IRA Income',
    # 'All Other Income',
    'Interest',
    # 'Interest Paid',
    'Dividends',
    # 'Charitable Contributions',
    # 'Medical & Dental Expenses',
    COL_TARGET,
    # 'Federal Amount of NY Adjusted Gross Income',
    # 'New York State Amount of NY Adjusted Gross Income',
    # 'New York Standard Deductions Claimed',
    # 'New York Itemized Deductions Claimed',
    # 'Deductions Used',
    # 'Dependent Exemptions Claimed',
    # 'Dependent Exemptions Used',
    # 'Taxable Income',
    # 'Tax Before Credits',
    # 'Tax Liability',  # the payment owed by an entity to a federal, state, or local tax authority
    # 'Tax Credits',  # money that taxpayers can subtract directly from the taxes they owe
    # 'Federal Adjustments',
    # 'Federal Adjusted Gross Income',
    # 'Additions State & Local Bond Interest',
    # 'Additions Public Employee Retirement System Contributions',
    # 'Additions College Choice Tuition Savings Program Distributions',
    # 'Other NY Additions',
    # 'Subtractions State & Local Income Tax Refunds',
    # 'Subtractions Government Pension Exclusion',
    # 'Subtractions Taxable Social Security Income',
    # 'Subtractions Federal Bond Interest',
    # 'Subtractions Pension & Annuity Exclusion',
    # 'Subtractions College Choice Tuition Savings Program Deductions/Earnings',
    # 'Other NY Subtractions',
    # 'Charitable Contributions',
    # 'Other Deductions',
    # 'Federal Deductions',
    # 'Income Taxes & Subtraction Adjustments',
    # 'NY Addition Adjustments',
    # 'NY Itemized Deduction Adjustment'
]

COLS2KEEP = [COL_TAX_YEAR, COL_INCOME_RANGE] + COLS_MONETARY


def get_score(regr, X, y):
    y_pred = regr.predict(X)
    score = r2_score(y, y_pred)
    print("score:", score)
    return score


def plot_predictions(regr, X, y):
    y_pred = regr.predict(X).flatten()
    print("number of samples:", len(y_pred))
    COL_SOURCE = "source"
    outputs1 = pd.DataFrame({COL_TARGET: y, COL_SOURCE: "actual"})
    outputs2 = pd.DataFrame({COL_TARGET: y_pred, COL_SOURCE: "predicted"})
    outputs = pd.concat([outputs1, outputs2])
    sns.displot(outputs, x=COL_TARGET, hue=COL_SOURCE, element="step")


# %% [markdown]
# Data Preprocessing: Cleaning, Feature Selection, Feature Engineering

# %%

def drop_rows_w_missing_values(df):
    nr_samples = len(df)
    df = df.dropna(axis="index", how="any")
    nr_samples_ = len(df)
    print(f"dropped {nr_samples - nr_samples_} samples with any missing values...")
    return df


path = CSV_FILE
df = pd.read_csv(path)
df[COL_TAX_YEAR] = df[COL_TAX_YEAR].astype(int)
df.fillna(0.0, inplace=True)

# Note: drop rows from aggregated columns
# TODO more exist:
for c in COLS_CATEGORICAL:
    df = df[df[c].apply(lambda cat: not "all" in cat.lower())]

print("remaining categories:")
for c in COLS_CATEGORICAL:
    print(c, df[c].unique())

for c in COLS_MONETARY:
    # average -> per person
    df[c] = df[c] / df[COL_CARDINALITY]
    # logarithm
    # TODO custom scaler into pipeline
    # Note: log(0) is undefined, so we add 1 to avoid this
    # Note no values in the range [0, 1] (mapped to negative logarithms)
    # Note: map negative values to negative logarithms
    df[c] = df[c].apply(lambda x: math.log(x + 1, 10) if x >= 0.0 else -math.log(-x, 10))

# TODO where do missing values come from?
df = drop_rows_w_missing_values(df)

# TODO only predict for people who pay taxes -> easier prediction, data less skewed
df = df[df[COL_TARGET] > 0.0]

df = df[COLS2KEEP]

df.to_csv(DATA_FOLDER / "cleaned.csv", index=False)

print("done.")
print(f"found {len(df)} rows, {len(df.columns)} columns\n")

# %% [markdown]
# Generate Basic Data Report

# %%

import ydata_profiling

profile = ydata_profiling.ProfileReport(df,
                                        minimal=False
                                        )
profile.to_file(REPORTS_FOLDER / "data_distribution_report.html")

# sns.pairplot(df[COLS_MONETARY].sample(100), diag_kind='kde')
# plt.tight_layout()  # prevent clipping
# plt.savefig(REPORTS_FOLDER/"column_pairplot.png", dpi=200)
# plt.close()


# %% [markdown]
# Split data into training and test set using the year column

# %%

COLS_INPUT = list(set(COLS_MONETARY) - {COL_TARGET})
X_train = df[df[COL_TAX_YEAR] == 2015]
X_test = df[df[COL_TAX_YEAR] == 2016]
y_train = X_train[COL_TARGET]
y_test = X_test[COL_TARGET]
X_train = X_train[COLS_INPUT]
X_test = X_test[COLS_INPUT]

# visualize distribution of target variable (train vs. test)
outputs1 = pd.DataFrame({COL_TARGET: y_train, COL_SOURCE: "training (year 2015)"})
outputs2 = pd.DataFrame({COL_TARGET: y_test, COL_SOURCE: "test (year 2016)"})
outputs = pd.concat([outputs1, outputs2])
sns.displot(outputs, x=COL_TARGET, hue=COL_SOURCE, element="step")

# %% [markdown]
# Fit and evaluate a linear model

# %%
from sklearn.linear_model import LinearRegression

regr = LinearRegression()
regr.fit(X_train, y_train)
print("score on test set:", get_score(regr, X_test, y_test))
plot_predictions(regr, X_test, y_test)

# %% [markdown]
# Fit and evaluate a simple single layer neural network

# %%
from sklearn.neural_network import MLPRegressor

regr = MLPRegressor(random_state=1, max_iter=500,
                    # hidden_layer_sizes=(100, 100, 100),
                    hidden_layer_sizes=(100,),
                    )
regr.fit(X_train, y_train)

# import joblib
# joblib.dump(regr, "model.joblib")

print("score on test set:", get_score(regr, X_test, y_test))
plot_predictions(regr, X_test, y_test)

# %% [markdown]
# Fit and evaluate a multi-layer neural network


# %% [markdown]
# pytorch (lightning) model

# %%
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

import collections
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks
from torch.utils.data import DataLoader, Dataset

NR_OUTP_FEATURES = 1

# DEBUG = True
DEBUG = False
NR_DEBUG_SAMPLES = 1000
BATCH_SIZE = 128
LEARNING_RATE = 1e-1  # TODO on plateau
OPT_CRIT_NAME = "val_loss"
OPT_MODE = "min"
EARLY_STOP_PATIENCE = 20
EARLY_STOP_THRESH = 0.0  # reached zero loss
EARLY_STOP_MIN_DELTA = 1e-3  # accept every improvement
BATCH_COUNT_EVAL = 10  # evaluating every n-th batch (gradient update) -> save checkpoint
MAX_EPOCHS = 300  # early stopping should kick in much earlier
LOSS_FCT = F.mse_loss

if DEBUG:
    print("INFO: DEBUG mode is on!")
    EARLY_STOP_PATIENCE = 1
    MAX_EPOCHS = 2

# Check that MPS is available
if torch.backends.mps.is_available():
    device_id = "mps"
else:
    if torch.backends.mps.is_built():
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )
    else:
        print("MPS not available because the current PyTorch install was not " "built with MPS enabled.")
    device_id = "cuda" if torch.cuda.is_available() else "cpu"

print(f"pytorch: Using {device_id} device")
torch_device = torch.device(device_id)
torch.set_default_dtype(torch.float32)

early_stopping_cb = callbacks.EarlyStopping(
    monitor=OPT_CRIT_NAME,
    mode=OPT_MODE,
    min_delta=EARLY_STOP_MIN_DELTA,
    # number of allowed validation checks with no improvement
    patience=EARLY_STOP_PATIENCE,
    # verbose=False,
    stopping_threshold=EARLY_STOP_THRESH,
)

training_kws = dict(
    callbacks=[early_stopping_cb, ],
    check_val_every_n_epoch=1,
    # val_check_interval=BATCH_COUNT_EVAL,  # more frequent evaluation
    max_epochs=MAX_EPOCHS,
)
if device_id == "mps":  # Apple Silicon GPU
    training_kws.update(dict(accelerator='mps', devices=1))

trainer = pl.Trainer(**training_kws)


class DataFrameDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        super().__init__()
        self.X = X.values.astype(np.float32)
        self.y = y.values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # TODO attention. reset index!
        sample = self.X[idx]
        target = self.y[idx]
        return sample, target


class TorchNN(pl.LightningModule):
    def __init__(
            self,
            learning_rate: float,
            input_size: int,
            **kwargs,
    ):
        super().__init__()
        # automatically save all the hyperparameters passed to init
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_size = NR_OUTP_FEATURES

        # classification head
        layers = (
            # TODO hidden layers,...
            nn.Linear(input_size, self.output_size),
            nn.Linear(input_size, self.output_size),
        )
        self.nn = nn.Sequential(*layers)

    def get_activations(self, X: torch.Tensor) -> torch.Tensor:
        return self.nn(X)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # defines the prediction/inference actions
        return self.get_activations(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = torch.tensor(X.values, dtype=torch.float32)
        y = self.forward(X).detach().numpy()
        return y

    def get_loss(self, batch):
        X, targets = batch
        activations = self.get_activations(X)
        return LOSS_FCT(activations, targets)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. independent of forward()
        training_loss = self.get_loss(batch)
        self.log("train_loss", training_loss)
        return training_loss

    def validation_step(self, batch, batch_idx):
        # Note: evaluating a single batch -> the eval sat should only have 1 batch
        validation_loss = self.get_loss(batch).item()
        # print(f"\n{validation_loss=}")
        self.log(OPT_CRIT_NAME, validation_loss)

    def test_step(self, batch, batch_idx):
        # Note: evaluating a batch, but the eval sat only has 1 batch
        x_val, y_true = batch
        y_pred = self(x_val)
        # TODO evaluate and log metrics

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=True
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.reduce_lr_on_plateau,
                "monitor": OPT_CRIT_NAME,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
                "frequency": 1,
            }
        }


def get_ds_loader(X, y, batch_size, shuffle):
    dataset = DataFrameDataset(X, y)
    ds_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return ds_loader


def train_model():
    nr_input_features = X_train.shape[1]
    # data preparation
    # shuffle each iteration for randomized batches
    train_loader = get_ds_loader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    # only one batch
    eval_loader = get_ds_loader(X_test, y_test, batch_size=len(y_test), shuffle=False)

    regr = TorchNN(
        learning_rate=LEARNING_RATE,
        input_size=nr_input_features,
    )
    regr.to(torch_device)

    # fit the head
    trainer.fit(regr, train_loader, eval_loader)
    # TODO
    print("evaluating the evaluation set")
    trainer.test(regr, eval_loader)
    return regr


regr = train_model()
print("score on test set:", get_score(regr, X_test, y_test))
plot_predictions(regr, X_test, y_test)

# %% [markdown]
# tensorflow

# %%
from keras import regularizers
import tensorflow as tf

tf.random.set_seed(1)
print("TensorFlow version:", tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"{len(gpus)} GPUs Available:", gpus)

from tensorflow import keras
from tensorflow.keras import layers

LOSS = 'mean_squared_error'  # 'mean_absolute_error'
ACT_FCT = 'relu'
DROPOUT = 0.0
LAYER_SIZE = 20
NR_HIDDEN_LAYER = 5
WEIGHT_REGULARISER = None  # regularizers.l2(0.001)
ACTIVITY_REGULARIZER = None  # regularizers.l1(0.001)
batch_size = 128
lr = 1e-1

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(len(X_test))


def build_and_compile_model(norm):
    layer_list = [norm, ]
    # layer_list = []

    for i in range(NR_HIDDEN_LAYER):
        layer_list += [
            layers.Dense(LAYER_SIZE, activation=ACT_FCT, kernel_regularizer=WEIGHT_REGULARISER,
                         activity_regularizer=ACTIVITY_REGULARIZER
                         ),
            layers.BatchNormalization(),
        ]
        if DROPOUT > 0.0:
            layer_list += [layers.Dropout(DROPOUT)]
    layer_list += [
        layers.Dense(1,
                     # activation='linear',
                     activation=ACT_FCT,
                     ),
    ]  # target
    model = keras.Sequential(layer_list)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
    )
    model.compile(loss=LOSS,
                  optimizer=optimizer, )
    return model


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel(LOSS)
    plt.semilogy()
    plt.legend()
    plt.grid(True)


normalizer = tf.keras.layers.Normalization(axis=-1)
# fit the state of the preprocessing layer to the data by calling Normalization.adapt:
normalizer.adapt(X_train)

regr = build_and_compile_model(normalizer)
print(regr.summary())
print("training...")
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True,
                                            start_from_epoch=10)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=1 / 2,
    patience=2,
    verbose=1,
    mode='auto',
    min_delta=0.0,
    cooldown=0,
    min_lr=0.0,
)
callbacks = [callback, reduce_lr]
history = regr.fit(
    X_train,
    y_train,
    validation_data=test_dataset,
    verbose=0,
    callbacks=callbacks,
    epochs=300,  # max
)

# regr.save('dnn_model')
# reloaded = tf.keras.models.load_model('dnn_model')
# results = dnn_model.evaluate(test_features, test_labels, verbose=0)


print("score on test set:", get_score(regr, X_test, y_test))
plot_loss(history)

# %%

plot_predictions(regr, X_test, y_test)
# plot_predictions(regr, X_train, y_train)


# %% [markdown]
# Visualise the prediction errors

# %%

y_test_pred = regr.predict(X_test).flatten()
errors = y_test - y_test_pred
sns.displot(errors, element="step")

# %% [markdown]
# Reduce the dimensionality of the data and visualise it


# %%
import umap

DIM_COLS = [f"dim{i}" for i in [1, 2]]


def reduce_dimensionality(X, nr_dims: int, **h_params):
    print(f"UMAP {h_params=}")
    print(f"reducing dimensionality from {X.shape[1]} to {nr_dims}... ")
    dim_reducer = umap.UMAP(n_components=nr_dims, metric="euclidean", **h_params)
    X_reduced = dim_reducer.fit_transform(X).T
    print("done.")
    return X_reduced


def add_umap_cols(df, columns, **h_params):
    X_orig = df[columns].values
    nr_dims = 2
    X_reduced = reduce_dimensionality(X_orig, nr_dims=nr_dims, **h_params)
    for d, umap_emb in enumerate(X_reduced):
        df[DIM_COLS[d]] = umap_emb


params = dict(n_neighbors=5, min_dist=0.3)
add_umap_cols(df, COLS_MONETARY, **params)

# %%
sns.relplot(data=df, x=DIM_COLS[0], y=DIM_COLS[1], hue=COL_INCOME_RANGE)  # s=size, **kwargs)
