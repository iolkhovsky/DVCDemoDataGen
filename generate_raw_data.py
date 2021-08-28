import argparse
import numpy as np
from os import makedirs
from os.path import isdir, isfile, join
import pandas as pd
from sklearn.datasets import make_classification
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Toy generator of a raw dataset")
    parser.add_argument("--config", type=str, default="data_generator.yml",
                        help="Absolute path to source videofile")
    return parser.parse_args()
    
    
def read_yaml(path):
    assert isfile(path), f"{path} doesnt exist"
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        return data


def generate_data(args):
    config = read_yaml(args.config)
    print(f"Running raw data generation according settings from '{args.config}':\n")
    print(config)
    dataset_path = config["output"]
    if not isdir(dataset_path):
        makedirs(dataset_path)
    dataset_pars = config["dataset"]
    X, y = make_classification(**dataset_pars)
    data = np.hstack([X, np.expand_dims(y, axis=1)])
    columns = [f"feature_{i}" for i in range(config["dataset"]["n_features"])] + ["label"]
    df = pd.DataFrame(data=data, columns=columns)
    output_csv_path = join(dataset_path, "raw_data.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"Generated data saved to: {output_csv_path}")
    print("Data content:\n")
    print(df.head())


if __name__ == "__main__":
    generate_data(parse_args())
