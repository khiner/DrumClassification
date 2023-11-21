# This script generates a curated subset of `e-gmd-v1.0.0.csv`.
# It also excludes several kits to limit the sound range to "standard"-sounding drum kits.

import pandas as pd

DATASET_DIR = 'dataset/e-gmd-v1.0.0'
METADATA_PATH = f'{DATASET_DIR}/e-gmd-v1.0.0.csv'
SLIM_METADATA_OUT_CSV_PATH = 'dataset/e-gmd-v1.0.0-slim.csv'

if __name__ == '__main__':
    metadata_df = pd.read_csv(METADATA_PATH)
    original_row_count = len(metadata_df)
    all_kit_names = set(metadata_df.kit_name.dropna().unique())
    include_kit_names = set([
        'Acoustic Kit', 'Studio (Live Room)', 'Classic Rock', 'Jazz Funk',
        'ClassicMetal (80s-90s)', '60s Rock', 'Modern Funk', 'Dark Hybrid', 'Big Room (Layered)',
        'Fat Rock (Power Toms)', 'Pop-Rock (Studio)',
        'Dry & Heavy (Folk Rock)', 'Second Line', 'Heavy Metal', 'Arena Stage',
        'Warmer Funk', 'Super Boom (Layered)', 'Jazz',
        'Live Rock', 'Shuffle (Blues)', 'Alternative (METAL)', 'Rockin Gate (80s)',
        'West Coast (FUNK)', 'Live Fusion', 'Speed Metal',
        'Cassette (Lo-Fi Compress)', 'Bigga Bop (Jazz)', 'Funk Rock',
        'Alternative (Rock)', 'Tight Prog', 'Unplugged',
    ])
    # Just to be explicit about what we're keeping, and what we're leaving out:
    expected_exclude_kit_names = set([
        'JingleStacks (2nd Hi-Hat)', 'Raw Dnb (Layered Hybrid)', 'More Cowbell (Pop-Rock)',
        '808 Simple', '909 Simple', 'Nu RNB', 'Ele-Drum', 'Custom1', 'Custom2', 'Custom3',
        'Compact Lite (w/ Tambourine HH)', 'Deep Daft',
    ])
    exclude_kit_names = all_kit_names - include_kit_names

    print(f'Found {len(all_kit_names)} unique kits:')
    print(all_kit_names)

    print(f'Keeping {len(include_kit_names)} kits:')
    print(include_kit_names)

    print(f'Excluding {len(exclude_kit_names)} kits:')
    print(exclude_kit_names)

    assert(exclude_kit_names == expected_exclude_kit_names)

    original_session_row_count = len(metadata_df)
    metadata_df = metadata_df[metadata_df.kit_name.isin(include_kit_names)]
    print('')
    print(f'Saving slimmed dataset of {len(metadata_df)} rows to {SLIM_METADATA_OUT_CSV_PATH}.')
    print(f'Excluded {original_session_row_count - len(metadata_df)} of {original_row_count} rows.')
    metadata_df.to_csv(SLIM_METADATA_OUT_CSV_PATH, index=False)
