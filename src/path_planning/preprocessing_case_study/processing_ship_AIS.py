import pandas as pd
import os

# Steps:
# 1. Sort the data based on the 'Time' column.
# 2. Remove rows until the 'SOG' value is greater than zero.
# 3. Replace NaN in 'SOG' with 0.
# 4. Convert 'Time' column values to minutes.
# 5. Subtract the first row's 'Time (mins)' value from all rows to start from zero.
# 6. Convert 'Time (mins)' to integers by truncating.
# 7. Group rows by 'Time (mins)' and average the other columns.


def time_to_minutes(time_str):
        date_part, time_part = time_str.split('_')
        hours = int(time_part[:2])
        minutes = int(time_part[2:4])
        seconds = int(time_part[4:6])
        total_minutes = hours * 60 + minutes + seconds / 60.0
        return total_minutes

def process_csv(file_path: str):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['Millisecond', 'COG'], errors='ignore')

    df = df.sort_values(by='Time')

    df = df.dropna(how='any')

    df['SOG'] = df['SOG'].fillna(0)
    df = df[df['SOG'] > 0].reset_index(drop=True)

    df['Time (mins)'] = df['Time'].apply(time_to_minutes)
    df.drop('Time', axis=1, inplace=True)
    first_time = df['Time (mins)'].iloc[0]
    df['Time (mins)'] = df['Time (mins)'] - first_time

    df['Time (mins)'] = df['Time (mins)'].astype(int)
    df = df.groupby('Time (mins)', as_index=False).mean()

    # Extract the directory and filename, and prepare the new file path
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    new_filename = os.path.join(directory, f"processed_{filename}")

    # Export the DataFrame to a new CSV file in the same location
    df.to_csv(new_filename, index=False)

    print(f"File processed and saved as {new_filename}")

# Example usage
file_path = '/workspaces/MELO_workspace/src/path_planning/preprocessing_case_study/ship_states.csv'
process_csv(file_path)
