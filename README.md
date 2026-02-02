# Permutation Importance in AMT

The goal of this work is to use domain-specific permutations and maskings to determine the importance of harmonic features in Automatic Music Transcription. 

## Installation and deployment

To reproduce these experiments, clone the repository and place the datasets into the `datasets` directory. Note that you need to initialize the model submodules in order to access their frameworks. You can create a compatible conda environment using the configuration file `environment.yaml`. 

```bash
$ git clone https://github.com/Xenobii/pi_amt.git
$ git submodule update --init --recursive
$ conda env create -f environment.yaml
```

In order to replicaate the experiments, run the main pipeline using:

```bash
$ python -m pi_amt -m
```

This will automatically run all the experiments configured in `config/config.yaml`, for all models, datasets and permutations ($2\times2\times8=32$ runs).

For running a subset of experiments, you can run the same command while specifying any of the available models, datasets or permutations.

```bash
$ python -m pi_amt -m model=basic_pitch, dataset=maps, permutation=fund_mask, harm_mask
```


## Models

For this study we use two instrument-agnostic models for automatic music transcription. 

Model|Paper| Github Repository
:---:|:---:|:---:
Timbre-Trap | [Timbre-Trap: A low-resource framework for instrument-agnostic music transcription](https://arxiv.org/abs/2309.15717)|  [github](https://github.com/sony/timbre-trap)
Basic Pitch | [A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation](https://arxiv.org/abs/2203.09893) | [official github](https://github.com/spotify/basic-pitch.git), [pytorch implementation](https://github.com/gudgud96/basic-pitch-torch.git)

## Datasets

We evalutate on two large piano datasets: MAESTRO and MAPS.

 - [Download Meastro](https://magenta.withgoogle.com/datasets/maestro)
 - [Download MAPS](https://adasp.telecom-paris.fr/resources/2010-07-08-maps-database/)

## Permutations 

We define the following permutations:

Permutation|Intuition|Purpose
:---|:---|:---
Random|Random frequency bin swapping across entire spectrum F|Evaluate model resistance to noise
Microtonal|Random frequency bin swapping within the bins of a semitone| Evaluate importance of high frequency resolution
High-Frequency|Random frequency bin swapping for frequency bins $f>f_0$|Evaluate whether a model ignores high frequencies


## Masks

We define the following maskings:

Mask|Intuition|Purpose
:---|:---|:---
Fundamental masking|Mask the fundamental frequency of each note|Evaluate the model’s reliance on harmonics
Harmonic masking|Mask everything but the fundamental frequencies of each note|Evaluate the model’s reliance on the fundamental
Soft fundamental masking|Mask the fundamental frequency of each note|Evaluate the model’s reliance on harmonics (more stable)
Soft harmonic masking|Replace everything but the fundamental frequencies of each note with low-amplitude noise (more realistic)|Evaluate the model’s reliance on the fundamental (more stable)