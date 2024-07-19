import os
import pandas as pd
from collections import defaultdict

from utils import (
    set_pandas_options,
    get_umls_folder,
    find_matched_concept_pairs,
    find_atom_to_atom_relationships,
    build_concept_aui_dict,
)


def process_umls(processing_type="drug_interactions"):
    """
    Main function to process UMLS data for different types of interactions.

    Args:
    processing_type (str): Type of processing to perform. Options are "drug_interactions" or "drug_disease_interactions".

    This function orchestrates the entire UMLS processing pipeline, including:
    1. Setting up the environment and configurations
    2. Processing MRSTY file to get concept types
    3. Filtering MRCONSO file
    4. Building concept-AUI dictionary
    5. Finding matched concept pairs
    6. Finding atom-to-atom relationships
    """
    set_pandas_options()
    umls_folder = get_umls_folder()

    # Define file paths
    mrsty_file = os.path.join(umls_folder, "MRSTY.RRF")
    mrconso_file = os.path.join(umls_folder, "MRCONSO.RRF")
    mrrel_file = os.path.join(umls_folder, "MRREL.RRF")

    # Configuration for different processing types
    processing_configs = {
        "drug_interactions": {
            "output_folder": "drug_interactions",
            "term_types": {"T121", "T200"},  # Pharmacologic Substance, Clinical Drug
            "relationships": {
                "induced_by",
                "contraindicated_mechanism_of_action_of",
                "has_risk_factor",
                "may_be_prevented_by",
                "risk_factor_of",
                "may_prevent",
                "has_pharmacokinetics",
                "enzyme_metabolizes_chemical_or_drug",
                "used_by",
                "induces",
                "has_contraindicated_mechanism_of_action",
                "chemical_or_drug_plays_role_in_biological_process",
                "has_related_factor",
                "time_modifier_of",
                "negatively_regulates",
                "modified_by",
                "associated_with",
                "has_contraindicated_drug",
                "uses_substance",
                "has_excluded_associated_finding",
                "chemical_or_drug_is_metabolized_by_enzyme",
                "modifies",
                "has_contraindicated_class",
                "effect_may_be_inhibited_by",
                "clinically_associated_with",
                "has_contraindicated_physiologic_effect",
                "related_to",
                "substance_used_by",
                "may_inhibit_effect_of",
                "positively_regulates",
                "is_object_guidance_for",
                "uses",
                "contraindicated_physiologic_effect_of",
            },
        },
        "drug_disease_interactions": {
            "output_folder": "drug_disease_interactions",
            "term_types": {
                "drug": {"T121", "T200"},  # Pharmacologic Substance, Clinical Drug
                "disease": {
                    "T047",
                    "T048",
                    "T019",
                    "T046",
                    "T184",
                },  # Disease or Syndrome, Mental or Behavioral Dysfunction, Congenital Abnormality, Pathologic Function, Sign or Symptom
            },
            "relationships": {
                "contraindicated_with_disease",
                "has_contraindicated_drug",
                "induces",
                "clinically_associated_with",
                "has_contraindicated_class",
                "effect_may_be_inhibited_by",
                "associated_with",
            },
        },
        "drug_symptom_interactions": {
            "output_folder": "drug_symptom_interactions",
            "term_types": {
                "drug": {"T121", "T200"},  # Pharmacologic Substance, Clinical Drug
                "symptom": {
                    "T184",
                    "T033",
                    "T034",
                    "T048",
                    "T046",
                    "T201",
                },  # Sign or Symptom, Finding, Laboratory or Test Result, Mental or Behavioral Dysfunction, Pathologic Function, Clinical Attribute
            },
            "relationships": {
                "associated_with",
                "causes",
                "induces",
                "has_physiologic_effect",
                "clinically_associated_with",
                "has_side_effect",
                "has_adverse_effect",
                "has_contraindicated_physiologic_effect",
                "has_manifestation",
                "has_finding",
                "contraindicated_physiologic_effect_of",
                "time_modifier_of",
                "has_related_factor",
                "effect_may_be_inhibited_by",
                "may_be_prevented_by",
                "may_be_finding_of_drug_related_disorder",
            },
        },
    }

    # Set up output folder and file paths
    config = processing_configs[processing_type]
    output_folder = os.path.join(umls_folder, config["output_folder"])
    os.makedirs(output_folder, exist_ok=True)

    filtered_mrconso_file = os.path.join(output_folder, f"filtered_MRCONSO.txt")
    output_dict_file = os.path.join(output_folder, f"concept_aui_dict.txt")
    output_pairs_file = os.path.join(output_folder, f"matched_concept_pairs.txt")
    output_atoms_file = os.path.join(output_folder, f"atom_to_atom_relationships.txt")

    if processing_type in ["drug_disease_interactions", "drug_symptom_interactions"]:
        concept_types = process_mrsty_paired(
            mrsty_file, config["term_types"], output_folder
        )
    else:
        concept_types = process_mrsty(mrsty_file, config["term_types"], output_folder)

    filter_mrconso(mrconso_file, filtered_mrconso_file, concept_types)
    concept_aui_dict = build_concept_aui_dict(filtered_mrconso_file, output_dict_file)

    if processing_type in ["drug_disease_interactions", "drug_symptom_interactions"]:
        matched_pairs_df = find_matched_paired_concepts(
            mrrel_file,
            filtered_mrconso_file,
            output_pairs_file,
            config["relationships"],
            concept_types,
        )
    else:
        matched_pairs_df = find_matched_concept_pairs(
            mrrel_file,
            filtered_mrconso_file,
            output_pairs_file,
            config["relationships"],
        )

    print(f"Number of matched concept pairs: {len(matched_pairs_df)}")
    print(matched_pairs_df["relationship"].value_counts())

    atom_relationships_df = find_atom_to_atom_relationships(
        mrrel_file, filtered_mrconso_file, output_atoms_file, config["relationships"]
    )
    print(f"Number of atom-to-atom relationships: {len(atom_relationships_df)}")
    print(atom_relationships_df["relationship"].value_counts())


def process_mrsty(mrsty_file, desired_types, output_folder):
    """
    Process the MRSTY file to get concept types for drug interactions.

    Args:
    mrsty_file (str): Path to the MRSTY.RRF file
    desired_types (set): Set of desired semantic types
    output_folder (str): Path to the output folder

    Returns:
    dict: A dictionary mapping concept IDs to their semantic types
    """
    checkpoint_file = os.path.join(output_folder, f"concept_types_checkpoint.pkl")
    if os.path.exists(checkpoint_file):
        return pd.read_pickle(checkpoint_file)

    concept_types = {}

    with open(mrsty_file, "r", encoding="utf-8") as file:
        for line in file:
            fields = line.strip().split("|")
            concept_id = fields[0]
            concept_type = fields[1]
            if concept_type in desired_types:
                concept_types[concept_id] = concept_type

    pd.to_pickle(concept_types, checkpoint_file)
    return concept_types


def process_mrsty_paired(mrsty_file, desired_types, output_folder):
    """
    Process the MRSTY file to get concept types for paired interactions (e.g., drug-disease, drug-symptom).

    Args:
    mrsty_file (str): Path to the MRSTY.RRF file
    desired_types (dict): Dictionary of desired semantic types for each category
    output_folder (str): Path to the output folder

    Returns:
    dict: A dictionary mapping concept IDs to their category ('drug', 'disease', or 'symptom')
    """
    checkpoint_file = os.path.join(output_folder, f"concept_types_checkpoint.pkl")
    if os.path.exists(checkpoint_file):
        return pd.read_pickle(checkpoint_file)

    concept_types = {}

    with open(mrsty_file, "r", encoding="utf-8") as file:
        for line in file:
            fields = line.strip().split("|")
            concept_id = fields[0]
            concept_type = fields[1]
            for category, types in desired_types.items():
                if concept_type in types:
                    concept_types[concept_id] = category
                    break

    pd.to_pickle(concept_types, checkpoint_file)
    return concept_types


def filter_mrconso(mrconso_file, output_file, concept_types):
    """
    Filter the MRCONSO file to keep only the concepts of interest.

    Args:
    mrconso_file (str): Path to the MRCONSO.RRF file
    output_file (str): Path to the output filtered file
    concept_types (dict): Dictionary of concept types to keep
    """
    if os.path.exists(output_file):
        print(f"Filtered file {output_file} already exists.")
        return

    with open(mrconso_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            fields = line.strip().split("|")
            concept_id = fields[0]
            if concept_id in concept_types:
                outfile.write(line)

    print(f"Filtered entries have been written to {output_file}")


def find_matched_paired_concepts(
    mrrel_file,
    filtered_mrconso_file,
    output_pairs_file,
    specified_relationships,
    concept_types,
):
    """
    Find matched paired concepts (e.g., drug-disease, drug-symptom) from the MRREL file.

    Args:
    mrrel_file (str): Path to the MRREL.RRF file
    filtered_mrconso_file (str): Path to the filtered MRCONSO file
    output_pairs_file (str): Path to the output file for matched pairs
    specified_relationships (set): Set of relationships to consider
    concept_types (dict): Dictionary mapping concepts to their types

    Returns:
    pandas.DataFrame: DataFrame of matched paired concepts
    """
    if os.path.exists(output_pairs_file):
        print(f"Matched concept pairs file {output_pairs_file} already exists.")
        return pd.read_csv(
            output_pairs_file,
            sep="|",
            header=None,
            names=["concept1", "concept2", "relationship"],
        )

    matched_pairs = set()

    with open(mrrel_file, "r", encoding="utf-8") as infile:
        for line in infile:
            fields = line.strip().split("|")
            concept1 = fields[0]
            concept2 = fields[4]
            relationship = fields[7]

            if relationship in specified_relationships:
                if (
                    concept1 in concept_types
                    and concept2 in concept_types
                    and concept_types[concept1] != concept_types[concept2]
                ):
                    matched_pairs.add((concept1, concept2, relationship))

    with open(output_pairs_file, "w", encoding="utf-8") as outfile:
        for pair in matched_pairs:
            outfile.write(f"{pair[0]}|{pair[1]}|{pair[2]}\n")

    print(f"Matched paired concepts have been written to {output_pairs_file}")
    return pd.DataFrame(
        list(matched_pairs), columns=["concept1", "concept2", "relationship"]
    )


if __name__ == "__main__":
    for processing_type in [
        "drug_interactions",
        "drug_disease_interactions",
        "drug_symptom_interactions",
    ]:
        process_umls(processing_type)
        print("\n" * 2 + "==" * 20 + "\n" * 2)
