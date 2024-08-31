"""
Preprocesses the reports and projections csv
to filter and get the required data and writes
it into a csv file
"""
from collections import Counter
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split


CLASS_NAMES = [
    "normal",
    "atelectasis",
    "cardiomegaly",
    "infiltration",
    "nodule",
    "emphysema",
    "pleural thickening",
    "calcified granuloma",
    "opacity",
    "lung/hypoinflation",
    "thoracic vertebrae/degenerative",
    "spine/degenerative",
    "lung/hyperdistention",
    "daphragmatic eventration",
    "calcinosis"
]


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports-file",
        type=argparse.FileType("r"),
        default="data/indiana_reports.csv",
        help="the reports csv file")
    parser.add_argument(
        "--projections-file",
        type=argparse.FileType("r"),
        default="data/indiana_projections.csv",
        help="the projections csv file")
    parser.add_argument(
        "--train-output-path",
        type=argparse.FileType("w"),
        default="data/iu_train_data.csv",
        help="the labelled preprocessed train data output path")
    parser.add_argument(
        "--test-output-path",
        type=argparse.FileType("w"),
        default="data/iu_test_data.csv",
        help="the labelled preprocessed test data output path")
    parser.add_argument(
        "--val-output-path",
        type=argparse.FileType("w"),
        default="data/iu_val_data.csv",
        help="the labelled preprocessed validation data output path")
    parser.add_argument(
        "--output-path",
        type=argparse.FileType("w"),
        default="data/iu_data.csv",
        help="the labelled preproccesed data")
    args = parser.parse_args()
    return args


def preprocess(args: argparse.Namespace):
    reports = pd.read_csv(args.reports_file.name)
    projections = pd.read_csv(args.projections_file.name)
    xray_data = pd.merge(reports, projections)
    xray_data = xray_data.dropna(subset=["findings", "MeSH"])
    xray_data = xray_data[xray_data["projection"] == "Frontal"]
    assert isinstance(xray_data, pd.DataFrame)
    result = pd.DataFrame()
    i = 0
    normal_count = 0
    for _, row in xray_data.iterrows():
        classes = list(filter(lambda name: name in str(row["MeSH"]).lower(), CLASS_NAMES))
        if len(classes) == 0:
            continue

        condition = classes[0]
        if condition == "normal":
            normal_count += 1
            if normal_count > 300:
                continue

        result = pd.concat([result, pd.DataFrame.from_records([{"filename": row["filename"], "condition": condition, "findings": row["findings"]}], index=[i])])
        i += 1

    train_data, test_data = train_test_split(result, test_size=0.3)
    assert isinstance(train_data, pd.DataFrame) and isinstance(test_data, pd.DataFrame)
    test_data, val_data = train_test_split(test_data, test_size=0.33)
    assert isinstance(test_data, pd.DataFrame) and isinstance(val_data, pd.DataFrame)
    [data.reset_index(drop=True, inplace=True) for data in [train_data, test_data, val_data]]
    train_data.to_csv(args.train_output_path)
    test_data.to_csv(args.test_output_path)
    val_data.to_csv(args.val_output_path)
    result.to_csv(args.output_path)


if __name__ == "__main__":
    args = parse()
    preprocess(args)
