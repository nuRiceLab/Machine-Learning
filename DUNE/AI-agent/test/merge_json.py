import glob
import json
import pandas as pd

# Method 1: Simple concatenation (merging lists)
'''
all_data = []
for f in glob.glob("*.json"):
    with open(f, 'r') as infile:
        all_data.extend(json.load(infile))

with open("merged_list.json", "w") as outfile:
    json.dump(all_data, outfile)
'''
# Method 2: Using pandas (for dataframes)
all_files = glob.glob("*.json")
df_list = []
for file in all_files:
    df_list.append(pd.read_json(file))

merged_df = pd.concat(df_list, ignore_index=True)
merged_df.to_json("merged_dataframe.json", orient='records', indent=2)
