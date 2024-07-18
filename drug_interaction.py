import os
from collections import defaultdict
import pandas as pd


def set_pandas_options():
    pd.set_option("display.max_columns", 500)


def get_umls_folder():
    return "../cache/datasets/UMLS/META/"


def get_column_info():
    columns = [
        "CUI",
        "LAT",
        "TS",
        "LUI",
        "STT",
        "SUI",
        "ISPREF",
        "AUI",
        "SAUI",
        "SCUI",
        "SDUI",
        "SAB",
        "TTY",
        "CODE",
        "STR",
        "SRL",
        "SUPPRESS",
        "CVF",
    ]
    dtypes = {col: str for col in columns}
    return columns, dtypes


def load_or_create_conso_en(umls_folder, columns, dtypes):
    conso_en_path = os.path.join(umls_folder, "conso_en.rrf")

    if os.path.exists(conso_en_path):
        return pd.read_csv(
            conso_en_path,
            sep="|",
            header=None,
            names=columns,
            dtype=dtypes,
            index_col=False,
        )
    else:
        mrconso_path = os.path.join(umls_folder, "MRCONSO.RRF")
        chunks = pd.read_csv(
            mrconso_path,
            sep="|",
            header=None,
            names=columns,
            dtype=dtypes,
            chunksize=10000,
            index_col=False,
        )

        with open(conso_en_path, "w") as f_out:
            for chunk in chunks:
                conso_en_chunk = chunk[chunk["LAT"] == "ENG"]
                conso_en_chunk.to_csv(
                    f_out, sep="|", header=False, index=False, mode="a"
                )

        return pd.read_csv(
            conso_en_path,
            sep="|",
            header=None,
            names=columns,
            dtype=dtypes,
            index_col=False,
        )


def filter_conso_en(conso_en, sab_tty_dict):
    filter_series = pd.Series([False] * len(conso_en))
    for sab, tty_list in sab_tty_dict.items():
        filter_series |= (conso_en["SAB"] == sab) & (conso_en["TTY"].isin(tty_list))
    return conso_en[filter_series]


def filter_relations(relations_file, output_file, relation_categories, category):
    if os.path.exists(output_file):
        return pd.read_csv(
            output_file,
            sep="|",
            header=None,
            names=[
                "CUI1",
                "AUI1",
                "STYPE1",
                "REL",
                "CUI2",
                "AUI2",
                "STYPE2",
                "RELA",
                "RUI",
                "SRUI",
                "SAB",
                "SL",
                "RG",
                "DIR",
                "SUPPRESS",
                "CVF",
            ],
            index_col=False,
        )

    chunks = pd.read_csv(
        relations_file,
        sep="|",
        header=None,
        names=[
            "CUI1",
            "AUI1",
            "STYPE1",
            "REL",
            "CUI2",
            "AUI2",
            "STYPE2",
            "RELA",
            "RUI",
            "SRUI",
            "SAB",
            "SL",
            "RG",
            "DIR",
            "SUPPRESS",
            "CVF",
        ],
        chunksize=10000,
        index_col=False,
    )

    with open(output_file, "w") as f_out:
        for chunk in chunks:
            rel_filtered_chunk = chunk[
                chunk["RELA"].isin(relation_categories[category])
            ]
            rel_filtered_chunk.to_csv(
                f_out, sep="|", header=False, index=False, mode="a"
            )

    return pd.read_csv(
        output_file,
        sep="|",
        header=None,
        names=[
            "CUI1",
            "AUI1",
            "STYPE1",
            "REL",
            "CUI2",
            "AUI2",
            "STYPE2",
            "RELA",
            "RUI",
            "SRUI",
            "SAB",
            "SL",
            "RG",
            "DIR",
            "SUPPRESS",
            "CVF",
        ],
        index_col=False,
    )


def join_relations_concepts(relations_df, concepts_df, batch_size=10000):
    def join_batch(batch):
        # print(f"Batch shape: {batch.shape}")
        joined = batch.merge(
            concepts_df,
            left_on="AUI1",
            right_on="AUI",
            how="left",
            suffixes=("", "_concept1"),
        )
        # print(f"Joined shape after first merge: {joined.shape}")
        aui_col = "AUI_concept1" if "AUI_concept1" in joined.columns else "AUI"
        mask = pd.isna(joined[aui_col])
        # print(f"Mask shape: {mask.shape}, True values: {mask.sum()}")
        if mask.any():
            joined = joined.reset_index(drop=True)
            mask = mask.reset_index(drop=True)
            cui_join = joined.loc[mask].merge(
                concepts_df,
                left_on="CUI1",
                right_on="CUI",
                how="left",
                suffixes=("", "_concept1_cui"),
            )
            joined.update(cui_join)
        # print(f"Joined shape after CUI1 merge: {joined.shape}")
        joined = joined.merge(
            concepts_df,
            left_on="AUI2",
            right_on="AUI",
            how="left",
            suffixes=("", "_concept2"),
        )
        # print(f"Joined shape after AUI2 merge: {joined.shape}")
        aui_col2 = "AUI_concept2" if "AUI_concept2" in joined.columns else "AUI"
        mask = pd.isna(joined[aui_col2])
        # print(f"Mask shape: {mask.shape}, True values: {mask.sum()}")
        if mask.any():
            joined = joined.reset_index(drop=True)
            mask = mask.reset_index(drop=True)
            cui_join = joined.loc[mask].merge(
                concepts_df,
                left_on="CUI2",
                right_on="CUI",
                how="left",
                suffixes=("", "_concept2_cui"),
            )
            joined.update(cui_join)
        # print(f"Final joined shape: {joined.shape}")

        return joined

    result_chunks = []
    for start in range(0, len(relations_df), batch_size):
        end = start + batch_size
        batch = relations_df.iloc[start:end]
        result_chunks.append(join_batch(batch))
    return pd.concat(result_chunks, ignore_index=True)


def create_drug_subset(df, drug_cuis):
    """
    Create a subset of the main DataFrame containing only drug-drug interactions.
    """
    # Create a mask for rows where both concepts are drugs
    drug_mask = df["CUI1"].isin(drug_cuis) & df["CUI2"].isin(drug_cuis)

    # Apply the mask to create the drug subset
    drug_subset = df[drug_mask].reset_index(drop=True)

    return drug_subset


def filter_relations_df(df):
    """
    Create a readable version of the DataFrame.
    """
    filtered_df = pd.DataFrame(
        {"Concept1": df["STR"], "Relation": df["RELA"], "Concept2": df["STR_concept2"]}
    )
    filtered_df = filtered_df.dropna()
    return filtered_df.reset_index(drop=True)


def load_drug_cuis(mrsty_file, drug_semantic_types):
    """
    Load drug CUIs from MRSTY.RRF file based on specified semantic types.
    """
    columns = ["CUI", "TUI", "STN", "STY", "ATUI", "CVF"]
    dtypes = {
        "CUI": str,
        "TUI": str,
        "STN": str,
        "STY": str,
        "ATUI": str,
        "CVF": str,  # Changed to str to avoid potential parsing issues
    }

    mrsty_df = pd.read_csv(
        mrsty_file,
        sep="|",
        header=None,
        names=columns,
        usecols=["CUI", "TUI"],
        dtype=dtypes,
        index_col=False,
        quoting=3,  # This disables quoting, which can cause issues with '|' separators
    )

    drug_cuis = set(mrsty_df[mrsty_df["TUI"].isin(drug_semantic_types)]["CUI"])
    return drug_cuis


if __name__ == "__main__":
    set_pandas_options()
    umls_folder = get_umls_folder()
    print("Loading UMLS data from:", umls_folder)

    columns, dtypes = get_column_info()
    conso_en = load_or_create_conso_en(umls_folder, columns, dtypes)
    print("Loaded UMLS MRCONSO data:")

    sab_tty_dict = {
        "MSH": ["BD", "PN"],
        "RXNORM": ["IN", "BN"],
    }
    conso_en_filtered = filter_conso_en(conso_en, sab_tty_dict)
    print(conso_en_filtered.head())

    relation_categories = {
        "contraindication": [
            "induced_by",
            "contraindicated_mechanism_of_action_of",
            "has_risk_factor",
            "may_be_prevented_by",
            "risk_factor_of",
            "may_prevent",
            "used_by",
            "induces",
            "has_contraindicated_mechanism_of_action",
            "time_modifier_of",
            "negatively_regulates",
            "modified_by",
            "associated_with",
            "has_contraindicated_drug",
            "uses_substance",
            "chemical_or_drug_is_metabolized_by_enzyme",
            "modifies",
            "has_contraindicated_class",
            "effect_may_be_inhibited_by",
            "clinically_associated_with",
            "substance_used_by",
            "may_inhibit_effect_of",
            "positively_regulates",
            "uses",
            "contraindicated_physiologic_effect_of",
        ],
    }
    relations_file = os.path.join(umls_folder, "MRREL.RRF")
    output_file = os.path.join(umls_folder, "contraindication_rels.rrf")
    category = "contraindication"

    rel_filtered = filter_relations(
        relations_file, output_file, relation_categories, category
    )
    print("Filtered relations:")
    print(rel_filtered.head())

    df = join_relations_concepts(rel_filtered, conso_en)
    print("Joined relations and concepts:")
    print(df.shape)

    readable_df = filter_relations_df(df)
    print("Full Readable DataFrame:")
    print(readable_df.head())
    print(f"Number of total interactions: {len(readable_df)}")

    ##### Create Drug-Drug Interaction Subset #####
    # This is all interactions where both cuis fall in to T121 or T200

    # Creating drug subset
    mrsty_file = os.path.join(umls_folder, "MRSTY.RRF")
    drug_semantic_types = ["T121", "T200"]

    drug_cuis = load_drug_cuis(mrsty_file, drug_semantic_types)
    print(f"Number of drug CUIs loaded: {len(drug_cuis)}")

    # Create drug subset from the main DataFrame
    drug_subset = create_drug_subset(df, drug_cuis)
    print(f"Number of drug-drug interactions: {len(drug_subset)}")

    # Create readable version of the drug subset
    readable_drug_subset = filter_relations_df(drug_subset)

    print("\nDrug-Drug Interaction Subset (Readable):")
    print(readable_drug_subset.head())
    print(f"Number of readable drug-drug interactions: {len(readable_drug_subset)}")
    print(readable_drug_subset["Relation"].value_counts())

    ##### Create Drug-Disease Interaction Subset #####

    ##### Create Drug-Symptom Relation Subset #####
