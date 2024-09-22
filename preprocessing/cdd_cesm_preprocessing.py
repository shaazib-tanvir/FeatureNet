import docx
import os
import os.path
import argparse
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/cdd-cesm-images",
        help="the path for images")
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="data/cdd-cesm-reports",
        help="the path containing the report files in docx format")
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/cdd-cesm-annotations.xlsx",
        help="the xlsx file containing the annotations")
    parser.add_argument(
        "--view-type",
        type=str,
        default="DM",
        help="the type of view to use (DM/CESM)")
    parser.add_argument(
        "--train-output-path",
        type=argparse.FileType("w"),
        default="data/cdd_cesm_train_data.csv",
        help="the labelled preprocessed train data output path")
    parser.add_argument(
        "--test-output-path",
        type=argparse.FileType("w"),
        default="data/cdd_cesm_test_data.csv",
        help="the labelled preprocessed test data output path")
    parser.add_argument(
        "--val-output-path",
        type=argparse.FileType("w"),
        default="data/cdd_cesm_val_data.csv",
        help="the labelled preprocessed validation data output path")
    parser.add_argument(
        "--output-path",
        type=argparse.FileType("w"),
        default="data/cdd_cesm_data.csv",
        help="the labelled preproccesed data")
    return parser.parse_args()

def preprocess(args):
    data = pd.read_excel(args.annotations)
    annotations = pd.DataFrame()
    i = 0
    for _, row in data.iterrows():
        if row["Type"] == args.view_type:
            continue

        patient_number = row["Patient_ID"]
        image_name = row["Image_name"]
        classification = row["Pathology Classification/ Follow up"]
        file_path = os.path.join(args.reports_dir, f"P{patient_number}.docx")
        if not os.path.exists(file_path):
            continue

        document = docx.Document(os.path.join(args.reports_dir, f"P{patient_number}.docx"))
        report = " ".join([paragraph.text.rstrip() for paragraph in document.iter_inner_content() if paragraph.text != ""][2:])
        image_path = os.path.join(args.image_dir, f"{image_name.rstrip()}.jpg")
        annotations = pd.concat([annotations, pd.DataFrame.from_records([{"image_path": image_path, "classification": classification, "report": report}], index=[i])])
        i += 1

    train_data, test_data = train_test_split(annotations, test_size=0.2)
    test_data, val_data = train_test_split(test_data, test_size=0.5)

    [data.reset_index(drop=True, inplace=True) for data in [train_data, test_data, val_data]]
    train_data.to_csv(args.train_output_path)
    test_data.to_csv(args.test_output_path)
    val_data.to_csv(args.val_output_path)
    annotations.to_csv(args.output_path)


if __name__ == "__main__":
    args = parse()
    preprocess(args)
