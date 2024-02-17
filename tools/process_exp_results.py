import sys

import fire
import pandas as pd
import yaml
from tabulate import tabulate


def load_and_filter_csv(csv_input, metrics_yaml):
    # Load the CSV file
    df = pd.read_csv(csv_input)

    # Load the metrics dictionary from the YAML file
    with open(metrics_yaml, "r") as file:
        metrics_dict = yaml.safe_load(file)

    # Create an empty DataFrame to store the filtered data
    filtered_df = pd.DataFrame()

    # Iterate over the items in the metrics dictionary
    for dataset, metrics in metrics_dict.items():
        # Filter the DataFrame for the current dataset
        dataset_df = df[df["Dataset-name"] == dataset]

        # Keep only the columns present in the metrics list along with 'Experiment-series', 'Dataset-name' and 'Experiment-name'
        columns_to_keep = [
            "Experiment-series",
            "Dataset-name",
            "Experiment-name",
        ] + metrics
        dataset_df = dataset_df[columns_to_keep]

        # Append the filtered DataFrame to the final DataFrame
        filtered_df = filtered_df.append(dataset_df)

    # Replace missing values with 'TBA'
    filtered_df.fillna("TBA", inplace=True)

    return filtered_df


def generate_table(df, key_terms_yaml):
    # Load the key terms from the YAML file
    with open(key_terms_yaml, "r") as file:
        key_terms = yaml.safe_load(file)

    # Filter the DataFrame based on the key terms
    df = df[df["Experiment-name"].str.contains("|".join(key_terms))]

    # Convert the DataFrame to a LaTeX table
    latex_table = tabulate(
        df, tablefmt="latex", headers="keys", showindex=False
    )

    return latex_table


def main(csv_input, metrics_yaml, key_terms_yaml_1, key_terms_yaml_2):
    # Load and filter the CSV file or standard input
    if csv_input == "-":
        df = load_and_filter_csv(sys.stdin, metrics_yaml)
    else:
        df = load_and_filter_csv(csv_input, metrics_yaml)

    # Generate the LaTeX tables
    table_1 = generate_table(df, key_terms_yaml_1)
    table_2 = generate_table(df, key_terms_yaml_2)

    print(table_1)
    print(table_2)


if __name__ == "__main__":
    fire.Fire(main)
