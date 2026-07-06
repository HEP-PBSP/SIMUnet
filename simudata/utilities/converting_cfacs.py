import csv
import os
import re
from pathlib import Path
from conversor import _autoname
from yaml import safe_load
import shutil
# Configuration
CSV_FILE_PATH = "data_maps.csv"  # Replace with your actual CSV file path
TARGET_DIR = Path(
    "/Users/ellacole/codes/simunet/simunet_nnpdf/simudata/commondata"
)
CFACTORS_DIR = Path(
    "/Users/ellacole/miniconda3/envs/simunet_mac/share/NNPDF/data/theory_270/cfactor")
NEW_CFAC_DIR = TARGET_DIR / "cfactors"
# Regex compiler required for the autoname logic
_tev_searcher = re.compile(r"^\d+TEV$")



def find_matching_directories(csv_path, search_base_dir):
    search_base_dir = Path(search_base_dir)

    if not search_base_dir.exists():
        print(f"Warning: Target directory does not exist: {search_base_dir}")
        return

    print(f"Scanning CSV for 'NOT FOUND' entries with non-empty column 3...")
    print(f"Searching in: {search_base_dir}\n" + "-" * 50)

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = next(reader, None)  # Skip header row

        for row_idx, row in enumerate(reader, start=2):
            if not row or len(row) < 2:
                continue

            col1 = row[0].strip()
            col2 = row[1].strip()

            col3_has_value = len(row) >= 3 and row[2].strip() != ""
            cfactor_names = row[2].strip() if col3_has_value else None

            if cfactor_names is not None:
                cfactor_names = cfactor_names.strip("[]").replace("'", "").split()

            if col2.upper() in ["NOT FOUND", "NOT_FOUND"] and col3_has_value:
                print(f"Row {row_idx}: Old name '{col1}'")

                try:
                    # 1. Generate the autocomposed new name using the imported function
                    generated_new_name = _autoname(col1)

                    # 2. Extract the set_name directory target
                    set_folder_name = generated_new_name.rsplit("_", 1)[0]

                    # 3. Check if the directory exists
                    target_path = search_base_dir / set_folder_name

                    if target_path.is_dir():
                        print(f"  [✓] Found new format directory: {set_folder_name}")
                    else:
                        print(f"  [❌] Expected directory '{set_folder_name}' but it was not found")
                        raise NotImplementedError("Directory not found for the generated new name.")

                    #Read metadata file in the target_path
                    metadata_file = target_path / "metadata.yaml"
                    # import IPython; IPython.embed()  # Debugging breakpoint
                    with open(metadata_file, 'r') as f:
                        metadata = safe_load(f)
                        if len(metadata.get('implemented_observables'))>1:
                            obs_name = col1.split("_")[-1]
                            try:
                                obs = next(
                                    item
                                    for item in metadata.get("implemented_observables", [])
                                    if item.get("observable_name") == obs_name
                                )
                            except StopIteration:
                                print(f"  [❌] Observable '{obs_name}' not found in metadata for {set_folder_name}")
                            
                        else:
                            obs = metadata.get('implemented_observables', [])[0]

                        fk_names = obs['theory'].get('FK_tables', [])

                        if len(fk_names) > 1:
                            ends = ["_" + fk_name[0].split("_")[-1] for fk_name in fk_names]
                        else:
                            ends = ['']

                    for cfac_type in cfactor_names:
                        num =0
                        for end in ends:
                            # import IPython; IPython.embed()  # Debugging breakpoint
                            old_cfac_name = 'CF_' + cfac_type + '_' + col1 + end + '.dat'
                            old_cfac_path = CFACTORS_DIR / old_cfac_name
                            fk_name = fk_names[num][0]
                            num += 1
                            new_cfac_name = f"CF_{cfac_type}_{fk_name}.dat"
                            # Copy old cfactor file to new name in the target directory
                            # Make sure the target directory exists
                            NEW_CFAC_DIR.mkdir(parents=True, exist_ok=True)
                            new_cfac_path = NEW_CFAC_DIR / new_cfac_name
                            shutil.copy(old_cfac_path, new_cfac_path)
                            print(f"  [✓] Copied '{old_cfac_name}' to '{new_cfac_name}'")


                except NotImplementedError:
                    print(f"  [!] Skipped: Prefix not recognized by _autoname rules.")

    
if __name__ == "__main__":
    find_matching_directories(CSV_FILE_PATH, TARGET_DIR)