# This script searches through all MIDI files in the slimmed dataset to find all unique midi notes.
# (See `create_slim_metadata.py` for details on the slimmed dataset.)
#
# The following metadata output files are generated:
# * `dataset/note_occurrences_slim.csv`, with the following columns:
#   - `note`: The MIDI note number.
#   - `name`: The human-readable name for the note.
#   - `occurrences`: The number of times the a message with this note value occurs in the slimmed dataset.
#   - `rank`: The occurrence frequency rank of the note.
# * `dataset/label_mapping.csv`, with the following columns, generated from the top-5 most common notes.
#   This is the final note mapping used for training. Other notes are ignored.
#   - `id`: A unique ID for the note. Used for one-hot encoding.
#   - `note`: The MIDI note number.
#   - `name`: The human-readable name for the note.

from mido import MidiFile
import pandas as pd

# Relevant standard drum MIDI note mappings (there are higher values but they don't occur in the slimmed dataset)):
# There are two notes in the slimmed dataset with unknown mappings (below this standard range): 22 and 26.
# See https://computermusicresource.com/GM.Percussion.KeyMap.html and other sources.
NAME_FOR_NOTE = {
    35: 'Acoustic Bass Drum',
    36: 'Bass Drum',
    37: 'Side Stick',
    38: 'Acoustic Snare',
    39: 'Hand Clap',
    40: 'Electric Snare',
    41: 'Low Floor Tom',
    42: 'Closed Hi Hat',
    43: 'High Floor Tom',
    44: 'Pedal Hi-Hat',
    45: 'Low Tom',
    46: 'Open Hi-Hat',
    47: 'Low-Mid Tom',
    48: 'Hi-Mid Tom',
    49: 'Crash Cymbal 1',
    50: 'High Tom',
    51: 'Ride Cymbal 1',
    52: 'Chinese Cymbal',
    53: 'Ride Bell',
    54: 'Tambourine',
    55: 'Splash Cymbal',
    56: 'Cowbell',
    57: 'Crash Cymbal 2',
    58: 'Vibraslap',
    59: 'Ride Cymbal 2',
}

DATASET_DIR = 'dataset/e-gmd-v1.0.0'
METADATA_PATH = 'dataset/e-gmd-v1.0.0-slim.csv'
OCCURRENCES_OUT_CSV_PATH = 'dataset/note_occurrences_slim.csv'
MAPPING_OUT_CSV_PATH = 'dataset/label_mapping.csv'

# Output dataset column names:
RANK = 'rank'
NOTE = 'note'
NAME = 'name'
OCCURRENCES = 'occurrences'

# Output label mapping column names:
ID = 'id'
# NOTE
# NAME

if __name__ == '__main__':
    metadata_df = pd.read_csv(METADATA_PATH)
    all_notes = dict() # Key: MIDI note number, Value: Occurrence count
    unique_midi_filenames = metadata_df.midi_filename.unique()
    for midi_filename in unique_midi_filenames:
        midi_file_path = f'{DATASET_DIR}/{midi_filename}'
        print(f'Processing {midi_file_path}...')
        midi_file = MidiFile(midi_file_path)
        assert(len(midi_file.tracks) == 2) # First track is metadata, then one track of drum notes.
        for msg in midi_file.tracks[1]:
            if msg.type == 'note_on' and msg.velocity > 0:
                all_notes[msg.note] = all_notes.get(msg.note, 0) + 1

    print(f'{METADATA_PATH} has {len(all_notes)} unique notes with the following occurrence counts:')
    sorted_entries = sorted(all_notes.items(), key=lambda item: item[1], reverse=True)
    for key, value in sorted_entries:
        print(f"{key}: {value}")

    print(f'Writing note occurrence metadata to {OCCURRENCES_OUT_CSV_PATH}...')
    occurrences = {NOTE: [], NAME: [], OCCURRENCES: [], RANK: []}
    for i, (note, count) in enumerate(sorted_entries):
        occurrences[NOTE].append(note)
        occurrences[NAME].append(NAME_FOR_NOTE.get(note, 'Unknown'))
        occurrences[OCCURRENCES].append(count)
        occurrences[RANK].append(i)

    occurrences_df = pd.DataFrame(occurrences)
    occurrences_df.to_csv(OCCURRENCES_OUT_CSV_PATH, index=False)

    mapping = {ID: [], NOTE: [], NAME: []}
    for i, (note, count) in enumerate(sorted_entries[:5]):
        mapping[ID].append(i)
        mapping[NOTE].append(note)
        mapping[NAME].append(NAME_FOR_NOTE[note])

    mapping_df = pd.DataFrame(mapping)
    mapping_df.to_csv(MAPPING_OUT_CSV_PATH, index=False)
