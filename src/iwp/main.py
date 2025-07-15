import numpy as np
import scipy.sparse as sp
import os
import torch

from iwp.algorithms.algorithms import StronglyConvexNesterovAcceleratedGradientDescent 

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

    I = len(B_list)
    J, L = C.shape
    P = B_list[0].shape[1]

    blocks = []
    for i in range(I):
        row_blocks = [sp.csr_matrix((J, L))] * I + [sp.csr_matrix((J, P))]
        row_blocks[i] = sp.csr_matrix(C)
        blocks.append(sp.hstack(row_blocks, format='csr'))
    D = sp.vstack(blocks, format='csr') # shape: (I*J, I*L + P) if A is (L,L), B_i is (L,P)
    
    d = np.concatenate(d_list, axis=0)  # shape: (I*J,)

    row_blocks = []
    for i in range(I):
        blocks = [sp.csr_matrix((L, L))] * I + [B_list[i]]
        blocks[i] = sp.csr_matrix(A)
        row = sp.hstack(blocks, format='csr')
        row_blocks.append(row)
    E = sp.vstack(row_blocks, format='csr') # shape: (I*L, I*L + P)

    lambd = float(args.lambd)
    mu = float(args.mu)

    def f(x):
        Dx_minus_d = D @ x - d
        Ex = E @ x
        return (
            0.5 * np.vdot(Dx_minus_d, Dx_minus_d).real
            + 0.5 * lambd * np.vdot(Ex, Ex).real
            + 0.5 * mu * np.vdot(x, x).real
        )

    def df(x):
        Dx_minus_d = D @ x - d
        Ex = E @ x
        return (
            D.conj().T @ Dx_minus_d
            + lambd * E.conj().T @ Ex
            + mu * x
        )

    K_op = D.conj().T @ D + lambd * E.conj().T @ E + mu * sp.eye(I * L + P)
    K_eigenvalues = np.linalg.eigvals(K_op.toarray())
    K = np.max(np.abs(K_eigenvalues))

    algo = StronglyConvexNesterovAcceleratedGradientDescent(
        name=args.exp_name,
        f=f,
        df=df,
        K=K,
        mu=mu,
        logger=logger,
    )
    x_0 = np.zeros(I*L + P) # shape: (I*L + P,)
    algo.run(x0=x_0, max_iterations=1000)