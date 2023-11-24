# Drum classification

A drum classification model trained on Magenta's [Expanded Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/e-gmd).

Final project for Computational Data Analysis (CSE-6740).

## Training

### Download data

Download and extract the [full E-GMD dataset zip](https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip) into the `dataset/` directory at the repository root.

This zip includes all MIDI, audio, and metadata.
Your directory structure should look like this:

```
DrumClassification/
    dataset/
        e-gmd-v1.0.0/
            drummer1/
            drummer2/
            ...
            e-gmd-v1.0.0.csv
    ...
```

### Prepare data for training

This step is already done, resulting in the CSV files at the root of the `dataset/` directory.
However, if you would like to change any aspect of the data preprocessing, such as including a different set of drum kit types, the full data preprocessing pipeline can be run as follows:

_See the scripts for explanations of their roles in data preparation._

```shell
$ pip install pandas mido  # The only packages needed for preprocessing
$ python create_slim_metadata.py  # Outputs `dataset/e-gmd-v1.0.0-slim.csv`
$ python create_label_mapping  # Outputs `dataset/note_occurrences_slim.csv` and `label_mapping.csv`
$ python chop_dataset.py  # Outputs `dataset/chopped_raw.csv`
$ python clean_chopped_raw.py  # Outputs `dataset/chopped.csv`
```

### Explore the dataset

The `explore_dataset.ipynb` notebook provides a variety of data exploration tools, such as:

- visualizing relevant data distrubutions
- previewing the "chopped" drum hit clips
- listening to random supercuts of clips for all training labels

### Train

```shell
$ pip install torch torchaudio pandas
$ pip install SoundFile  # On Mac, torchaudio needs a backend to load wav files.
$ python train.py
```
