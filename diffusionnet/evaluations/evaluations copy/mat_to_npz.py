import os
import argparse
import scipy.io
import numpy as np

def convert_mat_to_npz(input_dir, target_dir):
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if not os.path.exists(input_dir):
        print("Input directory doesn't exist!")
        return

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".mat"):
            # Construct the full file path
            file_path = os.path.join(input_dir, filename)
            # Load the .mat file
            data = scipy.io.loadmat(file_path)

            # Construct the output file path
            output_file = os.path.join(target_dir, filename.replace('.mat', '.npz'))
            # Save the file in .npz format
            np.savez(output_file, **data)

    print("Conversion complete.")

def main():
    parser = argparse.ArgumentParser(description='Convert .mat files to .npz format.')
    parser.add_argument('input_dir', type=str, help='Directory containing .mat files')
    parser.add_argument('target_dir', type=str, help='Directory to save .npz files')

    args = parser.parse_args()

    convert_mat_to_npz(args.input_dir, args.target_dir)

if __name__ == "__main__":
    main()
