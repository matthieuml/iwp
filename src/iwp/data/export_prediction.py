def save_complex_vector(filename, x):
    with open(filename, "w") as f:
        # write dimensions
        f.write(f"{x.shape[0]} \n")
        # write real and imag parts side by side
        for complex in x:
            f.write(f"  ({complex.real:.6f},{complex.imag:.6f})\n")
        f.write("\n")