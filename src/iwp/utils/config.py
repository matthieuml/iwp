import argparse
import logging
import os

import yaml

logger = logging.getLogger("iwp")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the experiment.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"),
        help="Path to the config file",
    )

    return parser.parse_args()


def load_yaml_into_namespace(
    yaml_file: str, namespace: argparse.Namespace
) -> argparse.Namespace:
    """
    Load a YAML file and merge its content into the given argparse Namespace.

    Args:
        yaml_file (str): Path to the YAML file.
        namespace (argparse.Namespace): The current Namespace object.

    Returns:
        argparse.Namespace: Updated Namespace with the values from the YAML file.
    """
    # Load the YAML file
    with open(yaml_file, "r") as file:
        yaml_data = yaml.safe_load(file)  # Parse the YAML content as a dictionary

    # Merge the YAML data into the Namespace
    namespace_dict = vars(namespace)  # Convert Namespace to dictionary
    namespace_dict.update(yaml_data)  # Update with YAML data
    logger.debug(f"Loaded YAML file: {yaml_file} as config")
    return argparse.Namespace(**namespace_dict)  # Convert back to Namespace
