import pandas as pd
from itertools import product
from pathlib import Path
import logging
from funconnect.stats.variable import Variables
from funconnect.stats.quantity import Quantities

logger = logging.getLogger("funconnect")


CATEGORICAL = dict(
    hva=pd.CategoricalDtype(categories=["V1", "HVA"], ordered=True),
    layer=pd.CategoricalDtype(
        categories=["L2/3", "L4", "L5"], ordered=True
    ),  # WARNING:there are very few L6 presyns and they will be excluded
    brain_area=pd.CategoricalDtype(categories=["V1", "RL", "LM", "AL"], ordered=True),
    proj_hva=pd.CategoricalDtype(
        categories=["V1->V1", "HVA->HVA", "V1->HVA", "HVA->V1"], ordered=True
    ),
    hva_layer=pd.CategoricalDtype(
        categories=[
            f"{hva}{layer}"
            for hva, layer in product(["V1", "HVA"], ["L2/3", "L4", "L5"])
        ],
        ordered=True,
    ),
    population=pd.CategoricalDtype(
        categories=["Connected", "ADP", "Same region"], ordered=True
    ),
    proofread_status=pd.CategoricalDtype(
        categories=["whole_cell", "projection_only"], ordered=True
    ),
    variable_name=pd.CategoricalDtype(
        categories=[v.name for v in Variables],
        ordered=True,
    ),
    quantity_name=pd.CategoricalDtype(
        categories=[q.name for q in Quantities],
        ordered=True,
    ),
)

CATEGORICAL = dict(
    **CATEGORICAL,
    pre_layer=CATEGORICAL["layer"],
    post_layer=CATEGORICAL["layer"],
    pre_hva_layer=CATEGORICAL["hva_layer"],
    post_hva_layer=CATEGORICAL["hva_layer"],
    proj_hva_layer=pd.CategoricalDtype(
        categories=[
            f"{pre_hva_layer}->{post_hva_layer}"
            for pre_hva_layer, post_hva_layer in product(
                CATEGORICAL["hva_layer"].categories, CATEGORICAL["hva_layer"].categories
            )
        ],
        ordered=True,
    ),
)


# Function to convert string to interval
def string_to_interval(s):
    s = s.strip("()[]").split(",")
    return pd.Interval(float(s[0]), float(s[1]), closed="right")


INTERVAL = dict(
    bins=string_to_interval,
)


def set_dtype(df):
    df = df.copy()
    for col in df.columns:
        if col in CATEGORICAL:
            df[col] = df[col].astype(pd.StringDtype("pyarrow")).astype(CATEGORICAL[col])
            logger.info(f"Converted {col} to categorical")
        elif col in INTERVAL:
            df[col] = df[col].apply(INTERVAL[col])
            logger.info(f"Converted {col} to interval")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("int64")
            logger.info(f"Converted {col} to int64")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float64")
            logger.info(f"Converted {col} to float64")

    return df


def read_csv(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = set_dtype(df)
    return df
