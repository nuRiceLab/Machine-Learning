import pandas as pd
import argparse
import glob

def concatenate_pickles(input_folder, output_file):
    try:
        # Find all pickle files in the input folder
        pickle_files = glob.glob(f"{input_folder}/*.pkl")
        
        if not pickle_files:
            print("No pickle files found in the specified folder.")
            return
        
        dataframes = []
        for file in pickle_files:
            try:
                df = pd.read_pickle(file)
                dataframes.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}, skipping this file.")
        
        # Concatenate all DataFrames
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df.to_pickle(output_file)
            print(f"Successfully saved concatenated DataFrame to {output_file}")
        else:
            print("No valid DataFrames to concatenate.")
    
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate multiple pickle files into a single DataFrame.")
    parser.add_argument("input_folder", type=str, help="Folder containing pickle files")
    parser.add_argument("output_pickle", type=str, help="Output pickle file")

    args = parser.parse_args()
    concatenate_pickles(args.input_folder, args.output_pickle)
