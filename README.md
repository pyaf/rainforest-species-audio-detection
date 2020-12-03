# Rainforest Connection Species Audio Detection


## Approach

1. Build a baseline multi-class classification model... DONE.
2.

## ToDo:

1. Decide Set of Augmentations that can be used in this challenge.
    * Waveform augs from audiomentations + SpecAugment
    * MixUP, TimeDrop
2. Decide on schedulers:
    1. scheduler: warmup cosine


## Revelations:

1. In this competition, the evaluation only considers the relative ranking of the species in the prediction. Nothing else, not their probabilities. Just the relative ranking.
2. train_bmp/*.bmp has specs of single labels only. They are cropped versions of one species at a time.
3. librosa.load: To preserve the native sampling rate of the file, use ``sr=None``.

## Questions:

