# =============================================
# CSV Analyzer — Day 5 Project
# Author: Abdelrhman
# Date: June 2026
# =============================================

import csv
import argparse


def load_csv(filepath):
    """
    Loads a CSV file and returns headers and rows.

    Args:
        filepath: Path to the CSV file

    Returns:
        Tuple of (headers, rows) or (None, None) if error
    """
    try:
        with open(filepath, "r") as file:
            reader = csv.DictReader(file)
            headers = reader.fieldnames
            rows = list(reader)
        return headers, rows
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found!")
        return None, None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None


def get_summary(headers, rows):
    """
    Returns basic summary statistics about the CSV.

    Args:
        headers: List of column names
        rows: List of row dictionaries

    Returns:
        Dictionary with summary statistics
    """
    return {
        "total_rows": len(rows),
        "total_columns": len(headers),
        "columns": list(headers)
    }


def get_column_stats(rows, column):
    """
    Calculates statistics for a numeric column.

    Args:
        rows: List of row dictionaries
        column: Column name to analyze

    Returns:
        Dictionary with min, max, average — or None if not numeric
    """
    try:
        values = [float(row[column]) for row in rows]
        return {
            "min": min(values),
            "max": max(values),
            "average": sum(values) / len(values),
            "total": sum(values)
        }
    except ValueError:
        return None


def get_unique_values(rows, column):
    """
    Returns unique values in a text column.

    Args:
        rows: List of row dictionaries
        column: Column name to analyze

    Returns:
        List of unique values
    """
    values = [row[column] for row in rows]
    return list(set(values))


def display_report(headers, rows):
    """
    Displays a complete analysis report.

    Args:
        headers: List of column names
        rows: List of row dictionaries
    """
    print("\n" + "=" * 50)
    print("         CSV ANALYSIS REPORT")
    print("=" * 50)

    # Summary
    summary = get_summary(headers, rows)
    print(f"\n📊 SUMMARY")
    print(f"   Total rows:    {summary['total_rows']}")
    print(f"   Total columns: {summary['total_columns']}")
    print(f"   Columns: {', '.join(summary['columns'])}")

    # Analyze each column
    print(f"\n📈 COLUMN ANALYSIS")
    for column in headers:
        print(f"\n   [{column}]")
        stats = get_column_stats(rows, column)
        if stats:
            print(f"   Type:    Numeric")
            print(f"   Min:     {stats['min']}")
            print(f"   Max:     {stats['max']}")
            print(f"   Average: {stats['average']:.2f}")
            print(f"   Total:   {stats['total']}")
        else:
            unique = get_unique_values(rows, column)
            print(f"   Type:    Text")
            print(f"   Unique values: {sorted(unique)}")

    print("\n" + "=" * 50)
    print("         END OF REPORT")
    print("=" * 50 + "\n")


def main():
    """
    Main function — handles command line arguments and runs analysis.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Analyze any CSV file and display statistics"
    )
    parser.add_argument(
        "--file",
        help="Path to the CSV file to analyze",
        required=True
    )

    # Parse arguments
    args = parser.parse_args()

    # Load and analyze
    print(f"Loading file: {args.file}")
    headers, rows = load_csv(args.file)

    if headers is None:
        return

    display_report(headers, rows)


if __name__ == "__main__":
    main()