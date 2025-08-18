import os

import numpy as np
import scipy.sparse as sp
import torch

from iwp.algorithms.algorithms import (
    ConstrainedConvexForwardBackward,
    ConstrainedConvexGradientDescent,
    ConvexGradientDescent,
    ConvexNesterovAcceleratedGradientDescent,
    StronglyConvexNesterovAcceleratedGradientDescent,
)
from iwp.data.load_experiment_data import load_experiment_data
from iwp.utils.config import load_yaml_into_namespace, parse_arguments
from iwp.utils.logger import setup_logger
from iwp.utils.utils import copy_file, make_dirs, set_seed

if __name__ == "__main__":

    args = parse_arguments()
    args = load_yaml_into_namespace(args.config, args)

    # Paths to save the results
    args.exp_path, args.visualizations_path = make_dirs(
        os.path.join(args.exp_path, args.exp_name)
    )

    logger = setup_logger(
        name="iwp",
        log_file=(
            os.path.join(args.exp_path, f"{args.exp_name}.log")
            if args.save_logs
            else None
        ),
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
        blocks.append(sp.hstack(row_blocks, format="csr"))
    D = sp.vstack(
        blocks, format="csr"
    )  # shape: (I*J, I*L + P) if A is (L,L), B_i is (L,P)

    d = np.concatenate(d_list, axis=0)  # shape: (I*J,)

    row_blocks = []
    for i in range(I):
        blocks = [sp.csr_matrix((L, L))] * I + [B_list[i]]
        blocks[i] = sp.csr_matrix(A)
        row = sp.hstack(blocks, format="csr")
        row_blocks.append(row)
    E = sp.vstack(row_blocks, format="csr")  # shape: (I*L, I*L + P)
    E_star = E.conj().T

    lambd = float(args.lambd)
    mu = float(args.mu)

    K_eigenvalues_A = np.linalg.eigvals((A.conj().T @ A).toarray())
    K_A = np.max(np.abs(K_eigenvalues_A))
    logger.info(f"K_A: {K_A}")

    def J_1(x):
        Dx_minus_d = D @ x - d
        Ex = E @ x
        return (
            0.5 * np.vdot(Dx_minus_d, Dx_minus_d).real
            + 0.5 * lambd * np.vdot(Ex, Ex).real
            + 0.5 * mu * np.vdot(x, x).real
        )

    def dJ_1(x):
        Dx_minus_d = D @ x - d
        Ex = E @ x
        return D.conj().T @ Dx_minus_d + lambd * E_star @ Ex + mu * x

    K_op_J_1 = D.conj().T @ D + lambd * E_star @ E + mu * sp.eye(I * L + P)
    K_eigenvalues_J_1 = np.linalg.eigvals(K_op_J_1.toarray())
    K_J_1 = np.max(np.abs(K_eigenvalues_J_1))
    logger.info(f"K_J_1: {K_J_1}")

    def J_2(x, threshold=1e-6):
        Dx_minus_d = D @ x - d
        Ex = E @ x
        if np.linalg.norm(Ex) < threshold:
            return (
                0.5 * np.vdot(Dx_minus_d, Dx_minus_d).real
                + 0.5 * mu * np.vdot(x, x).real
            )
        else:
            return np.inf

    K_op_J_2 = D.conj().T @ D + mu * sp.eye(I * L + P)
    K_eigenvalues_J_2 = np.linalg.eigvals(K_op_J_2.toarray())
    K_J_2 = np.max(np.abs(K_eigenvalues_J_2))
    logger.info(f"K_J_2: {K_J_2}")

    def J_3(m):
        total = 0.0
        for i in range(I):
            CA_inv_Bi_m = C @ sp.linalg.spsolve(A, B_list[i] @ m)
            diff = CA_inv_Bi_m - d_list[i]
            total += 0.5 * np.vdot(diff, diff).real
        return total + 0.5 * mu * np.vdot(m, m).real

    Ainv = sp.linalg.inv(A.tocsc())
    K_op_J_3 = sum(
        B_i.conj().T @ Ainv.conj().T @ C.conj().T @ C @ Ainv @ B_i for B_i in B_list
    ) + mu * sp.eye(P)
    K_eigenvalues_J_3 = np.linalg.eigvals(K_op_J_3.toarray())
    K_J_3 = np.max(np.abs(K_eigenvalues_J_3))
    logger.info(f"K_J_3: {K_J_3}")

    x_0 = np.zeros(I * L + P)  # shape: (I*L + P,)

    algo = StronglyConvexNesterovAcceleratedGradientDescent(
        name=args.exp_name,
        f=J_1,
        df=dJ_1,
        K=K_J_1,
        mu=mu,
        logger=logger,
    )
    # algo.run(x0=x_0, max_iterations=100)

    algo = ConstrainedConvexForwardBackward(
        name=args.exp_name,
        f=J_2,
        D=D,
        D_star=D.conj().T,
        E=E,
        E_star=E_star,
        d=d,
        mu=mu,
        gamma=1 / K_J_2,
        lambd=1,
        logger=logger,
    )
    # algo.run(x0=x_0, max_iterations=100)
