import numpy as np

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------

def print_matrix(title, mat):
    """Print a matrix with a title and nicer formatting."""
    print(f"\n{title}:")
    print(np.array_str(mat, precision=3, suppress_small=True))

def get_float_input(prompt):
    """Repeatedly ask for a float until valid input is given."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("❌ Invalid input. Please enter a number (e.g., 5, -2.5).")

def get_int_input(prompt):
    """Repeatedly ask for an integer until valid input is given."""
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("❌ Invalid input. Please enter an integer.")

def input_matrix_from_rows(m, n, name):
    """
    Allow the user to enter matrix entries row by row.
    Each row can be entered as space‑separated numbers or one number at a time.
    Returns a NumPy matrix of shape (m, n).
    """
    print(f"\nEnter {name} matrix ({m} x {n}):")
    mat = []
    for i in range(m):
        row_vals = []
        row_input = input(f"Row {i+1}: ").strip()
        if row_input:  # if user provides a row at once
            parts = row_input.split()
            if len(parts) == n:
                try:
                    row_vals = [float(x) for x in parts]
                except ValueError:
                    print("   Invalid numbers. Falling back to individual entry.")
                    row_vals = []
        if not row_vals:  # fallback to one‑by‑one entry
            for j in range(n):
                val = get_float_input(f"   Enter element ({i+1},{j+1}): ")
                row_vals.append(val)
        mat.append(row_vals)
    return np.array(mat, dtype=float)

def input_constant_matrix(m):
    """Allow the user to enter the constant column B (m x 1)."""
    print(f"\nEnter Constant Matrix B ({m} x 1):")
    col_input = input("Enter all constants as space‑separated values: ").strip()
    if col_input:
        parts = col_input.split()
        if len(parts) == m:
            try:
                B = np.array([[float(x)] for x in parts])
                print("   Column entered successfully.")
                return B
            except ValueError:
                print("   Invalid numbers. Falling back to individual entry.")
    # Fallback: one by one
    B = []
    for i in range(m):
        val = get_float_input(f"Enter B[{i+1}][1]: ")
        B.append([val])
    return np.array(B, dtype=float)

def show_echelon_form(matrix, title):
    """Compute and display the row echelon form (REF) of a matrix."""
    # Make a copy to avoid modifying original
    ref_mat = matrix.astype(float).copy()
    rows, cols = ref_mat.shape
    r = 0
    for c in range(cols):
        # Find pivot row
        pivot = None
        for i in range(r, rows):
            if abs(ref_mat[i, c]) > 1e-10:
                pivot = i
                break
        if pivot is None:
            continue
        # Swap to current row
        if pivot != r:
            ref_mat[[r, pivot]] = ref_mat[[pivot, r]]
        # Normalize pivot row (optional, for clarity)
        # ref_mat[r] = ref_mat[r] / ref_mat[r, c]
        # Eliminate below
        for i in range(r+1, rows):
            factor = ref_mat[i, c] / ref_mat[r, c]
            ref_mat[i, c:] -= factor * ref_mat[r, c:]
        r += 1
    print(f"\n{title} (Row Echelon Form, approximate):")
    print(np.array_str(ref_mat, precision=3, suppress_small=True))

# -------------------------------------------------
# MAIN PROGRAM
# -------------------------------------------------

def main():
    print("\n==========================================")
    print("   MATRIX CONSISTENCY LEARNING SYSTEM")
    print("==========================================\n")

    # Step 1: Matrix dimensions
    m = get_int_input("Enter number of equations: ")
    n = get_int_input("Enter number of variables: ")
    if m <= 0 or n <= 0:
        print("❌ Dimensions must be positive. Exiting.")
        return

    # Step 2: Coefficient matrix A
    A = input_matrix_from_rows(m, n, "Coefficient")
    # Step 3: Constant matrix B
    B = input_constant_matrix(m)

    # Step 4: Augmented matrix
    AB = np.hstack((A, B.reshape(m, 1) if B.ndim == 1 else B))

    # Display all matrices
    print_matrix("COEFFICIENT MATRIX A", A)
    print_matrix("CONSTANT MATRIX B", B)
    print_matrix("AUGMENTED MATRIX [A|B]", AB)

    # Backend calculations
    rank_A = np.linalg.matrix_rank(A)
    rank_AB = np.linalg.matrix_rank(AB)

    # -------------------------------------------------
    # QUESTION 1: Rank of A
    # -------------------------------------------------
    print("\n==========================================")
    print("QUESTION 1")
    print("==========================================")
    attempts = 0
    while True:
        try:
            user_rank_A = int(input("Find rank of coefficient matrix A: "))
        except ValueError:
            print("❌ Please enter an integer.")
            continue

        if user_rank_A == rank_A:
            print("\n✅ Correct Answer!")
            break
        else:
            attempts += 1
            print("\n❌ Incorrect Answer.")
            if attempts == 1:
                print("💡 Hint: Convert matrix into echelon form (non‑zero rows).")
                show_echelon_form(A, "Coefficient Matrix A")
            elif attempts == 2:
                print("💡 Hint: Count non‑zero rows after row operations.")
            else:
                print(f"✅ The rank of A is {rank_A}.")
                break

    # -------------------------------------------------
    # QUESTION 2: Rank of Augmented Matrix
    # -------------------------------------------------
    print("\n==========================================")
    print("QUESTION 2")
    print("==========================================")
    attempts = 0
    while True:
        try:
            user_rank_AB = int(input("Find rank of augmented matrix [A|B]: "))
        except ValueError:
            print("❌ Please enter an integer.")
            continue

        if user_rank_AB == rank_AB:
            print("\n✅ Correct Answer!")
            break
        else:
            attempts += 1
            print("\n❌ Incorrect Answer.")
            if attempts == 1:
                print("💡 Hint: Include the constant column when reducing.")
                show_echelon_form(AB, "Augmented Matrix")
            elif attempts == 2:
                print("💡 Hint: Reduce the augmented matrix to echelon form.")
            else:
                print(f"✅ The rank of [A|B] is {rank_AB}.")
                break

    # -------------------------------------------------
    # Determine consistency & solution type
    # -------------------------------------------------
    if rank_A == rank_AB:
        consistency = "Consistent"
        if rank_A == n:
            solution_type = "Unique Solution"
        else:
            solution_type = "Infinite Solutions"
    else:
        consistency = "Inconsistent"
        solution_type = "No Solution"

    # -------------------------------------------------
    # QUESTION 3: Consistency
    # -------------------------------------------------
    print("\n==========================================")
    print("QUESTION 3")
    print("==========================================")
    attempts = 0
    while True:
        user_consistency = input("Is the system Consistent or Inconsistent? ").strip().lower()
        if user_consistency in ["consistent", "c", "yes", "y"]:
            user_consistency = "consistent"
        elif user_consistency in ["inconsistent", "i", "no", "n"]:
            user_consistency = "inconsistent"
        else:
            print("❌ Please answer 'Consistent' or 'Inconsistent'.")
            continue

        if user_consistency == consistency.lower():
            print("\n✅ Correct!")
            break
        else:
            attempts += 1
            print("\n❌ Incorrect.")
            if attempts == 1:
                print("💡 Hint: Compare ranks of A and [A|B].")
            elif attempts == 2:
                print("💡 Hint: If ranks are equal → consistent.")
            else:
                print(f"✅ Correct Answer: {consistency}")
                break

    # -------------------------------------------------
    # QUESTION 4: Type of solution
    # -------------------------------------------------
    print("\n==========================================")
    print("QUESTION 4")
    print("==========================================")
    print("\nOptions (enter number or name):")
    print("1. Unique Solution")
    print("2. Infinite Solutions")
    print("3. No Solution")

    attempts = 0
    while True:
        answer = input("\nYour answer: ").strip().lower()
        # Map various inputs to the canonical string
        if answer in ["1", "unique", "unique solution", "unique solution"]:
            user_choice = "unique solution"
        elif answer in ["2", "infinite", "infinite solutions", "infinitely many"]:
            user_choice = "infinite solutions"
        elif answer in ["3", "no", "none", "no solution"]:
            user_choice = "no solution"
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3 or the phrase.")
            continue

        if user_choice == solution_type.lower():
            print("\n✅ Correct!")
            break
        else:
            attempts += 1
            print("\n❌ Incorrect.")
            if attempts == 1:
                if consistency == "Inconsistent":
                    print("💡 Hint: Inconsistent systems have no solution.")
                elif rank_A == n:
                    print("💡 Hint: Rank equals number of variables.")
                else:
                    print("💡 Hint: Rank is less than number of variables.")
            elif attempts == 2:
                print(f"💡 Correct answer: {solution_type}")
            else:
                print(f"✅ Correct Answer: {solution_type}")
                break

    # -------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------
    print("\n==========================================")
    print("FINAL ANALYSIS")
    print("==========================================")
    print(f"\nRank of A          = {rank_A}")
    print(f"Rank of [A|B]      = {rank_AB}")
    print(f"\nSystem Type        : {consistency}")
    print(f"Solution Type      : {solution_type}")

    # -------------------------------------------------
    # THEOREM EXPLANATION
    # -------------------------------------------------
    print("\n==========================================")
    print("THEOREM EXPLANATION")
    print("==========================================")
    if rank_A == rank_AB == n:
        print("""
Since:
    rank(A) = rank([A|B]) = number of variables
the system is CONSISTENT and has a UNIQUE SOLUTION.
""")
    elif rank_A == rank_AB and rank_A < n:
        print("""
Since:
    rank(A) = rank([A|B]) < number of variables
the system is CONSISTENT and has INFINITE SOLUTIONS.
""")
    else:
        print("""
Since:
    rank(A) ≠ rank([A|B])
the system is INCONSISTENT and has NO SOLUTION.
""")

    print("\n==========================================")
    print("THANK YOU FOR USING THE SYSTEM")
    print("==========================================")

if __name__ == "__main__":
    main()