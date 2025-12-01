import pandas as pd
import glob
import os

# --- 1. Load ALL demographic files and concatenate ---
demographic_files = glob.glob("./demographics/*.csv")

demodfs = []
for f in demographic_files:
    df = pd.read_csv(f)
    demodfs.append(df)

demographics = pd.concat(demodfs, ignore_index=True)

# Optional: drop duplicates just in case
demographics = demographics.drop_duplicates(subset=["Participant id"])

# --- 2. Select only the columns you want ---
cols_to_keep = [
    "Participant id",
    "Age",
    "Sex",
    "Ethnicity simplified",
    "Country of birth",
    "Country of residence",
    "Nationality",
    "Language",
]

demographics = demographics[cols_to_keep]

def add_demographics(path, out):
    # --- 3. Merge with your experiment file ---
    experiment = pd.read_csv(path)

    merged = experiment.merge(
        demographics,
        left_on="prolific_id",
        right_on="Participant id",
        how="left"  # keeps all experiment trials
    )

    # Drop duplicate column if needed
    merged = merged.drop(columns=["Participant id"])

    merged.to_csv(out, index=False)

if __name__ == "__main__":
    add_demographics("combined_actions.csv", "combined_actions_with_demographics.csv")