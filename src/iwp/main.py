import os
import torch

from iwp.data.load_experiment_data import load_experiment_data

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
        log_to_console=args.log_to_console,
    )

    # Save the config file in the experiment folder
    copy_file(args.config, os.path.join(args.exp_path, "config.yaml"))

    # Setup device and seed
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
    if args.device not in ["cpu", "cuda"]:
        raise ValueError("Device must be either 'cpu' or 'cuda'.")
    logger.info(f"Using device: {args.device}")
    set_seed(args.seed)

    # Load data
    A, B_list, C, d_list, m = load_experiment_data(args.data_path)
    