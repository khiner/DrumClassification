# Drum classification

A drum classification model trained on Magenta's [Expanded Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/e-gmd).

Final project for Computational Data Analysis (CSE-6740).

## Training

### Download data

Create a `dataset/` directory at the repository root, and download and extract the [full E-GMD dataset zip](https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip) into it.

This zip includes the MIDI, audio, and metadata.

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
