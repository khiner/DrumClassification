# This script outputs a CSV for training over the Expanded Groove MIDI Dataset.
# It assumes the dataset has been downloaded and extracted to `dataset/e-gmd-v1.0.0`,
# and that the metadata file `dataset/e-gmd-v1.0.0.csv` is present.
# Columns:
#   - file_path: The path to the audio file, relative to `dataset/e-gmd-v1.0.0`, excluding the file extension.
#   - begin_frame: The start audio frame (sample index) of the drum hit.
#   - num_frames: The number of audio frames (samples) in the drum hit.
#   - labels: A space-separated list of one or more drum instrument labels, using the following mapping:
#     - 0: Kick, 1: Snare, 2: Hi-hat, 3: Tom, 4: Cymbal

from mido import MidiFile
import pandas as pd
import torchaudio

DATASET_DIR = 'dataset/e-gmd-v1.0.0'
METADATA_PATH = f'{DATASET_DIR}/e-gmd-v1.0.0.csv'
SAMPLE_RATE = 44_100

# Output dataset column names
FILE_PATH_COL = 'file_path'
BEGIN_FRAME_COL = 'begin_frame'
NUM_FRAMES_COL = 'num_frames'
LABELS_COL = 'labels'

# E-GMD metadata column names
DRUMMER = 'drummer'
SESSION = 'session'
ID = 'id'
STYLE = 'style'
BPM = 'bpm'
BEAT_TYPE = 'beat_type' # 'beat' or 'fill'
TIME_SIGNATURE = 'time_signature'
DURATION = 'duration'
SPLIT = 'split' # 'train', 'test', or 'validation'
MIDI_FILENAME = 'midi_filename'
AUDIO_FILENAME = 'audio_filename'
KIT_NAME = 'kit_name'

# Appends a row per "drum hit" to the provided `dataset`.
# `dataset` is expected to have the dataset columns (described above) as keys, with list values.
# A "drum hit" is a segment of audio that contains at least one drum instrument.
# `file_path` is relative to `DATASET_DIR`, excluding the file extension.
def append_records(dataset, session_metadata):
    file_path = session_metadata[AUDIO_FILENAME].split('.')[0]
    midi_file_path = f'{DATASET_DIR}/{session_metadata[MIDI_FILENAME]}'
    bpm = session_metadata['bpm']

    midi_file = MidiFile(midi_file_path)
    assert(len(midi_file.tracks) == 2) # First track is metadata, then one track of drum notes.
    midi_track = midi_file.tracks[1]

    frames_per_tick = (SAMPLE_RATE * 60) / (bpm * midi_file.ticks_per_beat) 

    # Find the drum hits.
    # Rules:
    #   - Drum hits do not overlap.
    #   - A drum hit always starts with a note onset.
    #   - A drum hit cannot start or end in the middle of a playing note.
    #   - A drum hit ends at one of the two following events (whichever comes first):
    #     - The _beginning_ of the first note _after_ the last note belonging to the current drum hit ends.
    #     - The end of the track.
    current_hit = None
    active_notes = set()

    # Adds `current_hit` to `dataset`.
    def add_hit(end_frame):
        if current_hit is not None:
            dataset[FILE_PATH_COL].append(file_path),
            dataset[BEGIN_FRAME_COL].append(current_hit[BEGIN_FRAME_COL]),
            dataset[NUM_FRAMES_COL].append(int(end_frame - current_hit[BEGIN_FRAME_COL])),
            dataset[LABELS_COL].append(current_hit[LABELS_COL])

    # [Roland TD-17 MIDI implementation spec](https://static.roland.com/assets/media/pdf/TD-17_MIDI_Imple_eng01_W.pdf)
    # It doesn't say this in the spec, but it seems that note off events are sent as note on events with velocity 0.
    total_ticks = 0
    for msg in midi_track:
        total_ticks += msg.time # Delta time
        is_note_on = msg.type == 'note_on' and msg.velocity > 0
        is_note_off = msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)
        if not is_note_on and not is_note_off:
            # todo handle hi-hat pedal CC
            continue # Metadata, program change, etc.
        note = msg.note
        frame = int(total_ticks * frames_per_tick) # Round down to nearest frame.

        if current_hit is None:
            assert(is_note_on)
            current_hit = {BEGIN_FRAME_COL: frame, LABELS_COL: set([note])}
        else:
            if is_note_on:
                if len(active_notes) > 0:
                    # I think I'm seeing cases where a note starts and stops within the duration of another note.
                    # Need to verify this and decide if that's ok.
                    # assert(note not in current_hit[LABELS_COL]) 
                    current_hit[LABELS_COL].add(note)
                else:
                    # Add the current hit and start a new one.
                    add_hit(frame - 1)
                    current_hit = {BEGIN_FRAME_COL: frame, LABELS_COL: set([note])}
        if is_note_on:
            assert(note not in active_notes)
            active_notes.add(note)
        elif is_note_off:
            assert(note in active_notes)
            active_notes.remove(note)

    # Add the final hit to the dataset.
    add_hit(session_metadata[DURATION] * SAMPLE_RATE)

def preview_record(record):
    wav_file_path = f'{DATASET_DIR}/{record[FILE_PATH_COL]}.wav'
    audio = torchaudio.load(wav_file_path, frame_offset=record[BEGIN_FRAME_COL], num_frames=record[NUM_FRAMES_COL]) 
    print(audio.count) # todo preview audio in a notebook

if __name__ == '__main__':
    metadata_df = pd.read_csv(METADATA_PATH)
    dataset = {FILE_PATH_COL: [], BEGIN_FRAME_COL: [], NUM_FRAMES_COL: [], LABELS_COL: []}
    for i, session_metadata in metadata_df.iterrows():
        append_records(dataset, session_metadata)
        break # todo just a single session for now
    
    csv_out_path = 'dataset/chopped.csv'
    df = pd.DataFrame(dataset)
    df[LABELS_COL] = df[LABELS_COL].apply(lambda x: ' '.join(map(str, x))) # Convert labels to a space-separated string.
    df.to_csv(csv_out_path, index=False)
