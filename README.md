# Permutation Importance in AMT

The goal of this work is to use explainability methods to determine the importance of harmonic features in Music Information Retrieval models fot Automatic Music Transcription.


## Permutations

Given an input feature $X [F, T]$ consisting of $x_{f, t}$ bins and an output frame-wise prediction $Y[P, T]$ consisting of $y_{p, t}$. We define a permutation between $n$ samples as:

$$
    \pi : \{1, 2, ..., n\} \rightarrow \{1, 2, ..., n\} 
$$

We will evaluate the following permutations:

### 1. Random spectral permutations

The random permutation of frequencies is essentially replacing the signal with noise. Using such a permutation sparingly can evaluate the model's robustness to noise and problems in the input audio.

$$
    P_{randf}: x_{f, t} = x_{\pi(f), t}
$$

### 2. Harmonic permutations

Given that the input spectrogram follows a linear frequency distribution (CQT), we can target specific harmonic distributions for permutation.

We sample a random frequency $f\in F$. Knowing that the frequency's harmonics appear in $2f, 3f, ..., hf$ frequencies, we define that set of frequencies as:
$$
   s_f = [x_f, x_{2f}, ... , x_{hf}] 
$$

We define as harmonic permutation as the shuffle of these frequencies. This is very important semantically, since AMT models generally learn to identify notes by their fundamental frequency as well as their harmonics.

$$
    P_{harmf}: s_f = \pi(s_f)
$$

### 3. Microtonal frequencies

Given that the input feature has a resolution $F = B \times P$, where $P$ is the number of output pitches and $B$ is the number of bins per semitone, we can assign $s_p$ the set of frequency bins that pairs with each semitone. 

$$
    s_p = [ x_{f-i,t}, x_{f-i+1, t}, ..., x_{f, t}, ..., x_{f+1, t} ], i \in \frac{B}{2}
$$

The semantic importance of this permutation is evaluating the importance of the size of $B$ and by extension the resolution of the feature across the spectral dimension.

$$
    P_{microf}: s_p = \pi(s_p)
$$

### 4. Edge case frequencies

A common problem for AMT models is the lack of training examples for extreme cases, like extremely high and low notes. This may lead the model to completely disregard such frequencies as noise. We can evaluate this by inputting random noise to the model and estimating the power of the noise across pitch classes in the output:

$$
    X_{noise} \sim \mathcal{N}_{f, t}
$$

We can compute the general weight of each frequcny bin $f\in F$ as follows: