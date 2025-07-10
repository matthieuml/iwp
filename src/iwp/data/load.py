from scipy.sparse import coo_matrix

def load_sparse_matrix(filename, is_complex=True):
    with open(filename, 'r') as f:
        # Skip comment lines
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of file")
            if line.strip() and not line.startswith('#'):
                break

        # Read header: rows, cols, nnz, ...
        header = line.strip().split()
        rows, cols, nnz = int(header[0]), int(header[1]), int(header[2])

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
                real, imag = val_str.strip('()').split(',')
                val = complex(float(real), float(imag))
            else:
                val = float(parts[2])
            row_idx.append(row)
            col_idx.append(col)
            data.append(val)

    return coo_matrix((data, (row_idx, col_idx)), shape=(rows, cols))