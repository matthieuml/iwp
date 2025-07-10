import os
import torch

from iwp.utils.config import parse_arguments, load_yaml_into_namespace
from iwp.utils.utils import make_dirs, copy_file, set_seed
from iwp.utils.logger import setup_logger


if __name__ == "__main__":

    args = parse_arguments()
    args = load_yaml_into_namespace(args.config, args)

    # Paths to save the results
    args.exp_path, args.visualizations_path = make_dirs(
        os.path.join(args.exp_path, args.exp_name)
    )

    logger = setup_logger(
        name="iwp",
        log_file=os.path.join(args.exp_path, f"{args.exp_name}.log")
        if args.save_logs
        else None,
        level=args.log_level,
        log_to_console=args.verbose,
    )

    # Save the config file in the experiment folder
    copy_file(args.config, os.path.join(args.exp_path, "config.yaml"))

    # Execute the experiment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    set_seed(args.seed)
    