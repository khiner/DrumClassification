# Reads in the raw chopped dataset and outputs a cleaned version by excluding clips
# belonging to files that have at least one clip that either begins after the file ends or doesn't peak until after 0.4 seconds.
# These are taken as signs of files with invalid MIDI timing.

import pandas as pd
import wave
import torchaudio

SAMPLE_RATE = 44_100
PEAK_CUTOFF_SECONDS = 0.4
PEAK_CUTOFF_FRAMES = int(PEAK_CUTOFF_SECONDS * SAMPLE_RATE)
CHOPPED_RAW_PATH = 'dataset/chopped_raw.csv'
CHOPPED_PATH = 'dataset/chopped.csv'
DATASET_DIR = 'dataset/e-gmd-v1.0.0'

def get_num_frames(wav_file_path):
    with wave.open(wav_file_path, 'rb') as wav_file:
        return wav_file.getnframes()

def ends_after_file_end(row):
    total_num_frames = get_num_frames(f'{DATASET_DIR}/{row.file_path}')
    return row.begin_frame >= total_num_frames

# Returns true if the first peak in the clip is after the provided cutoff.
def peaks_after(row, peak_cutoff_frames):
    waveform, sample_rate = torchaudio.load(f'{DATASET_DIR}/{row.file_path}', frame_offset=row.begin_frame, num_frames=row.num_frames)
    waveform = waveform[0] # Only one channel.
    waveform = waveform / waveform.abs().max() # Normalize.
    peaks = (waveform.abs() > 1 - 1e-6).nonzero().view(-1)
    return peaks.nelement() > 0 and peaks[0] >= peak_cutoff_frames

# Look for indicators of invalid MIDI timing.
def find_invalid_clips(df):
    invalid_clips = []
    for i, row in df.iterrows():
        print(f'Checking {i} of {len(df)}')
        if ends_after_file_end(row) or peaks_after(row, PEAK_CUTOFF_FRAMES):
            invalid_clips.append(row)
    return invalid_clips

if __name__ == '__main__':
    chopped_raw_df = pd.read_csv(CHOPPED_RAW_PATH)
    invalid_clips = find_invalid_clips(chopped_raw_df)
    invalid_files = set([row.file_path for row in invalid_clips])
    print(f'Found {len(invalid_clips)} invalid clips in {len(invalid_files)} files.')

    chopped_df = chopped_raw_df[~chopped_raw_df.file_path.isin(invalid_files)] 
    chopped_df.to_csv(CHOPPED_PATH, index=False)
    print(f'Deleted {len(chopped_raw_df) - len(chopped_df)} rows from {CHOPPED_RAW_PATH} and saved remaining {len(chopped_df)} rows to {CHOPPED_PATH}.')
