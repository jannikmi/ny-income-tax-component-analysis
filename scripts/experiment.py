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

# %%

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
import sklearn
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
    score = sklearn.metrics.r2_score(y, y_pred)
    print("score:", score)
    return score


def plot_predictions(regr, X,y):
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

# %%
from keras import regularizers
import tensorflow as tf

tf.random.set_seed(1)
print("TensorFlow version:", tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"{len(gpus)} GPUs Available:", gpus)

from tensorflow import keras
from tensorflow.keras import layers

LOSS = 'mean_squared_error' # 'mean_absolute_error'
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
    embeddings = df[columns].values
    nr_dims = 2
    umap_embds = reduce_dimensionality(embeddings, nr_dims=nr_dims, **h_params)
    for d, umap_emb in enumerate(umap_embds):
        df[DIM_COLS[d]] = umap_emb


params = dict(n_neighbors=5, min_dist=0.3)
add_umap_cols(df, COLS_MONETARY, **params)

# %%
sns.relplot(data=df, x=DIM_COLS[0], y=DIM_COLS[1], hue=COL_INCOME_RANGE)  # s=size, **kwargs)
