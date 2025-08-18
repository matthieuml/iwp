import logging

import numpy as np
from scipy.sparse import coo_matrix

logger = logging.getLogger("iwp")


def load_sparse_matrix(filename: str, is_complex: bool = True) -> coo_matrix:
    with open(filename, "r") as f:
        try:
            # Skip comment lines
            while True:
                line = f.readline()
                if not line:
                    raise ValueError("Unexpected end of file")
                if line.strip() and not line.startswith("#"):
                    break

            # Read header: rows, cols, nnz, ...
            header = line.strip().split()
            rows, cols, nnz = int(header[0]), int(header[1]), int(header[2])

            logger.debug(
                f"Loading sparse matrix from {filename}: {rows} rows, {cols} cols, {nnz} non-zeros..."
            )

            row_idx = []
            col_idx = []
            data = []
            for _ in range(nnz):
                line = f.readline()
                parts = line.strip().split()
                row = int(parts[0])
                col = int(parts[1])
                if is_complex:
                    val_str = parts[2]
                    real, imag = val_str.strip("()").split(",")
                    val = complex(float(real), float(imag))
                else:
                    val = float(parts[2])
                row_idx.append(row)
                col_idx.append(col)
                data.append(val)

            logger.debug("DONE")
        except Exception as e:
            logger.error(f"Error reading sparse matrix file {filename}: {e}")
            raise ValueError(f"Error reading sparse matrix file {filename}: {e}")

    return coo_matrix((data, (row_idx, col_idx)), shape=(rows, cols))


def load_vector(filename: str, is_complex: bool = True) -> np.ndarray:
    with open(filename, "r") as f:
        try:
            # Read header: length
            line = f.readline()
            header = line.strip().split()
            length = int(header[0])

            logger.debug(f"Loading vector from {filename}: length {length}...")

            data = []
            while len(data) < length:
                line = f.readline()
                if not line:
                    raise ValueError("Unexpected end of file")
                parts = line.strip().split()
                for part in parts:
                    if len(data) >= length:
                        raise ValueError("More data than expected in vector file")
                    val_str = part
                    if is_complex:
                        real, imag = val_str.strip("()").split(",")
                        val = complex(float(real), float(imag))
                    else:
                        val = float(val_str)
                    data.append(val)

            logger.debug("DONE")

        except Exception as e:
            logger.error(f"Error reading vector file {filename}: {e}")
            raise ValueError(f"Error reading vector file {filename}: {e}")

    return np.array(data, dtype=np.complex128 if is_complex else np.float64)
