def extract_zips(outer_file):
    import os
    import zipfile
    import pandas as pd
    from io import BytesIO

    # List to collect all data
    all_data = []

    # Loop over each top-level zip file
    for filename in os.listdir(outer_file):
        if filename.endswith(".zip"):
            zip_path = os.path.join(outer_file, filename)
            
            with zipfile.ZipFile(zip_path, 'r') as top_zip:  # Open top-level ZIP
                
                for nested_zip_name in top_zip.namelist():
                    if nested_zip_name.endswith(".zip"):
                        nested_zip_bytes = BytesIO(top_zip.read(nested_zip_name))  # Read nested zip into memory

                        with zipfile.ZipFile(nested_zip_bytes, 'r') as nested_zip:
                            
                            for csv_name in nested_zip.namelist():
                                if csv_name.endswith(".csv"):
                                    with nested_zip.open(csv_name) as csv_file:
                                        try:
                                            df = pd.read_csv(csv_file, skiprows=1, nrows=48, header=None)
                                            all_data.append(df)
                                        except Exception as e:
                                            print(f"Failed to read {csv_name} in {filename}: {e}")
    
    return all_data

def extract_wind():
    import pandas as pd

    all_data = extract_zips("wind_zips")

    cols = ['date', 'hour', 'wind_system', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'wind_coast', 'x', 'x', 'x', 
            'wind_south', 'x', 'x', 'x', 'wind_west', 'x', 'x', 'x', 'wind_north', 'x', 'x', 'x', 'x']

    # Combine all 48-row chunks into one DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.columns = cols
    final_df = final_df.loc[:, final_df.columns != 'x']

    print("Complied ERCOT Wind Generation data into a DataFrame")

    return final_df

def extract_solar():
    import pandas as pd

    all_data = extract_zips("solar_zips")

    cols = ['date', 'hour', 'solar_system', 'x', 'x', 'x', 'solar_centerwest', 'x', 'x', 'x', 'solar_northwest', 
            'x', 'x', 'x', 'solar_farwest', 'x', 'x', 'x', 'solar_fareast', 'x', 'x', 'x', 'solar_southeast', 
            'x', 'x', 'x', 'solar_centereast', 'x', 'x', 'x', 'x', 'x']

    # Combine all 48-row chunks into one DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.columns = cols
    final_df = final_df.loc[:, final_df.columns != 'x']

    print("Complied ERCOT Solar Generation data into a DataFrame")

    return final_df


        