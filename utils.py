import os
from collections import defaultdict
import pandas as pd


def set_pandas_options():
    """
    Set Pandas display options to show more columns.

    This function increases the maximum number of columns that Pandas will display,
    which is useful when working with wide datasets.
    """
    pd.set_option("display.max_columns", 500)


def get_umls_folder():
    """
    Get the path to the UMLS META folder.

    Returns:
    str: The path to the UMLS META folder.
    """
    return "../cache/datasets/UMLS/META/"


def find_matched_concept_pairs(
    mrrel_file, filtered_mrconso_file, output_pairs_file, specified_relationships
):
    """
    Find and save matched concept pairs based on specified relationships.

    This function reads the MRREL file, filters relationships based on the specified set,
    and saves matched concept pairs to a file. It only considers concepts that are present
    in the filtered MRCONSO file.

    Args:
    mrrel_file (str): Path to the MRREL.RRF file.
    filtered_mrconso_file (str): Path to the filtered MRCONSO file.
    output_pairs_file (str): Path to save the matched concept pairs.
    specified_relationships (set): Set of relationship types to consider.

    Returns:
    pandas.DataFrame: DataFrame containing the matched concept pairs and their relationships.
    """
    if os.path.exists(output_pairs_file):
        print(f"Matched concept pairs file {output_pairs_file} already exists.")
        return pd.read_csv(
            output_pairs_file,
            sep="|",
            header=None,
            names=["concept1", "concept2", "relationship"],
        )

    unique_concepts = set()
    with open(filtered_mrconso_file, "r", encoding="utf-8") as infile:
        for line in infile:
            fields = line.strip().split("|")
            concept_id = fields[0]
            unique_concepts.add(concept_id)

    matched_pairs = set()

    with open(mrrel_file, "r", encoding="utf-8") as infile:
        for line in infile:
            fields = line.strip().split("|")
            concept1 = fields[0]
            concept2 = fields[4]
            relationship = fields[7]

            if relationship in specified_relationships:
                if concept1 in unique_concepts and concept2 in unique_concepts:
                    matched_pairs.add((concept1, concept2, relationship))

    with open(output_pairs_file, "w", encoding="utf-8") as outfile:
        for pair in matched_pairs:
            outfile.write(f"{pair[0]}|{pair[1]}|{pair[2]}\n")

    print(f"Matched concept pairs have been written to {output_pairs_file}")
    return pd.DataFrame(
        list(matched_pairs), columns=["concept1", "concept2", "relationship"]
    )


def find_atom_to_atom_relationships(
    mrrel_file, filtered_mrconso_file, output_atoms_file, specified_relationships
):
    """
    Find and save atom-to-atom relationships based on specified relationships.

    This function reads the MRREL file, filters relationships based on the specified set,
    and saves matched atom pairs to a file. It only considers atoms that are present
    in the filtered MRCONSO file.

    Args:
    mrrel_file (str): Path to the MRREL.RRF file.
    filtered_mrconso_file (str): Path to the filtered MRCONSO file.
    output_atoms_file (str): Path to save the matched atom pairs.
    specified_relationships (set): Set of relationship types to consider.

    Returns:
    pandas.DataFrame: DataFrame containing the matched atom pairs and their relationships.
    """
    if os.path.exists(output_atoms_file):
        print(f"Atom-to-atom relationships file {output_atoms_file} already exists.")
        return pd.read_csv(
            output_atoms_file,
            sep="|",
            header=None,
            names=["aui1", "aui2", "relationship"],
        )

    concept_aui_dict = defaultdict(list)
    with open(filtered_mrconso_file, "r", encoding="utf-8") as infile:
        for line in infile:
            fields = line.strip().split("|")
            concept_id = fields[0]
            aui = fields[7]
            concept_aui_dict[concept_id].append(aui)

    all_auis = {aui for auis in concept_aui_dict.values() for aui in auis}

    matched_atom_pairs = set()

    with open(mrrel_file, "r", encoding="utf-8") as infile:
        for line in infile:
            fields = line.strip().split("|")
            aui1 = fields[1]
            aui2 = fields[5]
            relationship = fields[7]

            if (
                relationship in specified_relationships
                and aui1 in all_auis
                and aui2 in all_auis
            ):
                matched_atom_pairs.add((aui1, aui2, relationship))

    with open(output_atoms_file, "w", encoding="utf-8") as outfile:
        for pair in matched_atom_pairs:
            outfile.write(f"{pair[0]}|{pair[1]}|{pair[2]}\n")

    print(
        f"Matched atom-to-atom relationships have been written to {output_atoms_file}"
    )
    return pd.DataFrame(
        list(matched_atom_pairs), columns=["aui1", "aui2", "relationship"]
    )


def build_concept_aui_dict(filtered_mrconso_file, output_dict_file):
    """
    Build a dictionary mapping concepts to their Atom Unique Identifiers (AUIs).

    This function reads the filtered MRCONSO file and creates a dictionary where
    each concept is mapped to a list of its associated AUIs. The dictionary is
    saved both as a text file and as a pickle file for later use.

    Args:
    filtered_mrconso_file (str): Path to the filtered MRCONSO file.
    output_dict_file (str): Path to save the concept-AUI dictionary.

    Returns:
    dict: A dictionary mapping concept IDs to lists of AUIs.
    """
    if os.path.exists(output_dict_file):
        print(f"Concept-AUI dictionary file {output_dict_file} already exists.")
        return pd.read_pickle(output_dict_file + ".pkl")

    concept_aui_dict = defaultdict(list)

    with open(filtered_mrconso_file, "r", encoding="utf-8") as infile:
        for line in infile:
            fields = line.strip().split("|")
            concept_id = fields[0]
            aui = fields[7]
            concept_aui_dict[concept_id].append(aui)

    with open(output_dict_file, "w", encoding="utf-8") as outfile:
        for concept_id, auis in concept_aui_dict.items():
            outfile.write(f"{concept_id}: {', '.join(auis)}\n")

    pd.to_pickle(dict(concept_aui_dict), output_dict_file + ".pkl")
    print(f"Concept-AUI dictionary has been written to {output_dict_file}")
    return concept_aui_dict
