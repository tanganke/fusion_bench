#!/usr/bin/env python3
"""
Script to convert CSV file to markdown format.
Numbers are multiplied by 100 and formatted to 2 decimal places.
Columns are sorted according to TASK_NAMES_TALL20 order.
"""

import os
import sys

import pandas as pd

# Define the task order from TASK_NAMES_TALL20
TASK_NAMES_TALL20 = [
    "sun397",
    "stanford-cars",
    "resisc45",
    "eurosat",
    "svhn",
    "gtsrb",
    "mnist",
    "dtd",
    "oxford_flowers102",
    "pcam",
    "fer2013",
    "oxford-iiit-pet",
    "stl10",
    "cifar100",
    "cifar10",
    "food101",
    "fashion_mnist",
    "emnist_letters",
    "kmnist",
    "rendered-sst2",
]


def csv_to_markdown(csv_file, output_file=None):
    """
    Convert CSV file to markdown format with numbers x100 and 2 decimal places.
    Columns are sorted according to TASK_NAMES_TALL20 order.

    Args:
        csv_file (str): Path to the input CSV file
        output_file (str, optional): Path to the output markdown file.
                                   If None, prints to stdout.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Create a copy for processing
    df_processed = df.copy()

    # Sort columns according to TASK_NAMES_TALL20 order
    # Start with model_name column, then add task columns in the specified order
    ordered_columns = ["model_name"]

    # Add columns that exist in both the CSV and TASK_NAMES_TALL20, in the TALL20 order
    for task_name in TASK_NAMES_TALL20:
        if task_name in df.columns:
            ordered_columns.append(task_name)

    # Add any remaining columns that aren't in TASK_NAMES_TALL20 (in case there are extras)
    for col in df.columns:
        if col not in ordered_columns:
            ordered_columns.append(col)

    # Reorder the dataframe columns
    df_processed = df_processed[ordered_columns]

    # Multiply numeric columns by 100 and format to 2 decimal places
    for column in df_processed.columns:
        if column != "model_name":  # Skip the model name column
            # Convert to numeric, multiply by 100, and format to 2 decimal places
            df_processed[column] = pd.to_numeric(df_processed[column], errors="coerce")
            df_processed[column] = (df_processed[column] * 100).round(2)
            # Format as strings with 2 decimal places
            df_processed[column] = df_processed[column].apply(lambda x: f"{x:.2f}")

    # Manually create markdown table for better formatting
    markdown_lines = []

    # Header row
    headers = list(df_processed.columns)
    header_line = "| " + " | ".join(headers) + " |"
    markdown_lines.append(header_line)

    # Separator row
    separator_parts = []
    for header in headers:
        if header == "model_name":
            separator_parts.append(":----------")
        else:
            separator_parts.append("----------:")
    separator_line = "| " + " | ".join(separator_parts) + " |"
    markdown_lines.append(separator_line)

    # Data rows
    for _, row in df_processed.iterrows():
        row_values = []
        for col in headers:
            row_values.append(str(row[col]))
        row_line = "| " + " | ".join(row_values) + " |"
        markdown_lines.append(row_line)

    markdown_content = "\n".join(markdown_lines)

    if output_file:
        with open(output_file, "w") as f:
            f.write(markdown_content)
        print(f"Markdown table saved to: {output_file}")
    else:
        print(markdown_content)


def main():
    if len(sys.argv) < 2:
        print("Usage: python csv_to_markdown.py <csv_file> [output_file]")
        print(
            "Example: python csv_to_markdown.py results/vit-b-32.csv results/vit-b-32.md"
        )
        sys.exit(1)

    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found.")
        sys.exit(1)

    try:
        csv_to_markdown(csv_file, output_file)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
