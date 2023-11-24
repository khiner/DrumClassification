# Reads in the raw chopped dataset and outputs a cleaned version.

import pandas as pd
import wave

CHOPPED_RAW_PATH = 'dataset/chopped_raw.csv'
CHOPPED_PATH = 'dataset/chopped.csv'
DATASET_DIR = 'dataset/e-gmd-v1.0.0'

def get_num_frames(wav_file_path):
    with wave.open(wav_file_path, 'rb') as wav_file:
        return wav_file.getnframes()

def find_invalid_clips(df):
    invalid_clips = []
    for _, row in df.iterrows():
        total_num_frames = get_num_frames(f'{DATASET_DIR}/{row.file_path}')
        if row.begin_frame + row.num_frames > total_num_frames:
            invalid_clips.append(row)
    return invalid_clips

if __name__ == '__main__':
    chopped_raw_df = pd.read_csv(CHOPPED_RAW_PATH)
    invalid_clips = find_invalid_clips(chopped_raw_df)
    invalid_files = set([row.file_path for row in invalid_clips]) # All files that contain invalid clips likely have invalid MIDI timing.
    chopped_df = chopped_raw_df[~chopped_raw_df.file_path.isin(invalid_files)] 
    chopped_df.to_csv(CHOPPED_PATH, index=False)
    print(f'Found {len(invalid_clips)} invalid clips in {len(invalid_files)} files.')
    print(f'Deleted {len(chopped_raw_df) - len(chopped_df)} rows from {CHOPPED_RAW_PATH} and saved remaining {len(chopped_df)} rows to {CHOPPED_PATH}.')
