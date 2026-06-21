---
title: Zapp-tain America - Audio Identifier
emoji: 🎵
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
---

# 🎵 Zapp-tain America — Audio Identifier
[Try the App](https://ee200-course-project-d7vldiwaojzjkdfnpf2xgh.streamlit.app/#zapp-tain-america-audio-identifier)

A from-scratch, **Shazam-style audio fingerprinting and song-identification system**, wrapped in an interactive Streamlit web app. Upload a short clip of a song and the app tells you which track in its library the clip came from — and shows you every intermediate step it used to reach that decision.

Built as the **EE200 course project** (Q3B).

---

## What it is

Music-recognition services like Shazam can name a song from a few noisy seconds recorded on a phone. They do this not by comparing raw audio (which is far too fragile to noise, volume, and compression) but by reducing each song to a compact, robust **acoustic fingerprint** and matching fingerprints instead of waveforms.

This project is a complete, transparent implementation of that idea. It indexes a library of songs into a searchable fingerprint database, and then identifies unknown clips against that database. Unlike a black-box service, it exposes the whole pipeline visually — the spectrogram, the constellation of detected peaks, and the voting histogram that ultimately decides the match — so you can *see* why a clip was matched to a particular song.

The technique follows the landmark-based fingerprinting approach of Wang (2003) and Ellis (2009): **constellation peaks → combinatorial hashing → offset-histogram voting**.

---

## How it works

The full identification pipeline, from raw audio to a named song:

```
  audio  →  spectrogram  →  peak constellation  →  paired hashes
                                                        │
  winner  ←  offset histogram  ←  matched DB hashes  ←──┘
```

### 1. Spectrogram

The clip is resampled to **mono at 11,025 Hz** and converted into a magnitude spectrogram using a Short-Time Fourier Transform (1024-sample windows, 256-sample hops ≈ 23 ms per frame). Magnitudes are converted to decibels and normalized so the loudest bin sits at 0 dB. This turns the audio into a time–frequency image of where its energy lives.

### 2. Constellation map (peak picking)

Only the most prominent points in that image are kept. A time–frequency bin is selected as a **peak** if it is the local maximum within a 19×19 neighborhood and is louder than −55 dB. Amplitude is then thrown away — only the `(frequency, time)` coordinates remain. The result is a sparse "constellation" of landmark points. These peaks are what survive background noise, volume changes, and lossy compression, which is exactly what makes the fingerprint robust.

### 3. Combinatorial hashing

Single peaks are not distinctive enough on their own, so peaks are combined. Each peak acts as an **anchor** and is paired with up to 8 nearby peaks in a "target zone" ahead of it in time (1–40 frames away, within 80 frequency bins). Each anchor–target pair is packed into a compact **32-bit integer hash** encoding `(frequency₁, frequency₂, time-gap)`. These paired hashes are highly specific, dramatically reducing accidental collisions between unrelated songs.

### 4. Inverted-index database

Every song in the library is fingerprinted the same way, and all hashes are stored in an **inverted index** that maps each hash → the list of `(song, time-position)` pairs where it occurs. This index ships pre-built as `database.pkl` (**50 songs, ~1.5 million distinct hashes**), so the app can identify clips immediately on launch with no indexing step required.

### 5. Matching by offset histogram

To identify a query clip, every one of its hashes is looked up in the index. For each matching database hash, the app records the **time offset** between where the hash appears in the database song and where it appears in the query.

The key insight: if the clip really is an excerpt of a song, then *many* of its hashes will share the **same offset**, because the clip is simply a time-shifted slice of the original. The app builds a per-song histogram of these offsets — the song whose histogram has the single tallest bar wins, and the height of that bar is the **match score**. A genuine match scores in the hundreds or thousands of aligned hashes; an unrelated clip produces only a handful of coincidental matches. If no song clears the configurable threshold, the clip is reported as **"not in the database."**

---

## Features

- **Single-clip mode** — identify one uploaded clip and visualize the entire pipeline: its spectrogram, its constellation of detected peaks, and the offset histogram that decided the winning match — shown alongside the predicted song, its score, and a ranked table of the top candidate matches.
- **Batch mode** — identify many clips at once and download a `results.csv` file with exactly two columns, `filename,prediction`.
- **Adjustable match threshold** — a slider sets the minimum number of aligned hashes required to count as a match, letting you trade off between rejecting unknown songs (raise it) and catching faint matches (lower it).
- **Fully self-contained** — ships with its pre-indexed database, so it works the moment it is deployed.

---

## The song database

The fingerprint library lives in `database.pkl`, which contains **50 indexed songs** and roughly **1.5 million distinct hashes**. The app loads it once (cached for speed) and lists every indexed song in an expandable panel in the UI. Because the database is pre-built and bundled with the app, no separate indexing run is needed before identifying clips.

---

## Project structure

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web UI — single-clip and batch modes, plus all the visualizations. |
| `fingerprint.py` | The fingerprinting and matching engine: spectrogram, peak picking, combinatorial hashing, the `FingerprintDB` inverted index, and audio loading. |
| `database.pkl` | The pre-indexed song library (50 songs, ~1.5M hashes). |
| `requirements.txt` | Python dependencies. |
| `packages.txt` | System packages (`ffmpeg`, `libsndfile1`) for audio decoding. |
| `Dockerfile` | Container build for Hugging Face Spaces and other hosts. |
| `README.md` | This file. |

---

## Running locally

```bash
# 1. System audio libraries (Debian / Ubuntu)
sudo apt-get install ffmpeg libsndfile1

# 2. Python dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

Then open the local URL Streamlit prints (by default <http://localhost:8501>), choose a mode, and upload a clip.

---

## Deployment

The repository is ready to deploy as a **Hugging Face Space** using the Docker SDK — the YAML block at the very top of this file configures the Space (title, hardware, and the `app_port` Streamlit listens on). The included `Dockerfile` installs the required system audio libraries (`ffmpeg`, `libsndfile1`) so decoding works out of the box. The same `Dockerfile` is portable to other container hosts such as Render, Fly.io, or Railway.

---

## Configuration & tuning

The algorithm's behavior is controlled by a small set of constants at the top of `fingerprint.py`:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `SR` | 11025 | Sample rate (Hz) all audio is resampled to. |
| `N_FFT` | 1024 | STFT window length (samples). |
| `HOP` | 256 | STFT hop length (≈ 23 ms per frame). |
| `PEAK_NEIGH_T` / `PEAK_NEIGH_F` | 19 / 19 | Local-maximum neighborhood (time × frequency) for peak picking. |
| `PEAK_MIN_DB` | −55 | Peaks quieter than this (relative to the loudest bin) are ignored. |
| `FAN_OUT` | 8 | Maximum target peaks paired with each anchor. |
| `DT_MIN` / `DT_MAX` | 1 / 40 | Min/max time gap (frames) between an anchor and its targets. |
| `DF_MAX` | 80 | Maximum frequency-bin difference within a target zone. |

At runtime, the **match-threshold slider** in the UI controls the minimum aligned-hash score required to accept a match versus reporting "not in the database."

---

## Supported audio formats

`wav`, `mp3`, `m4a`, `flac`, and `ogg`. WAV/FLAC/OGG are decoded directly via `soundfile`; compressed formats such as MP3/M4A fall back to `ffmpeg`.

---

## References

1. A. L.-C. Wang, *"An Industrial-Strength Audio Search Algorithm,"* ISMIR 2003.
2. D. P. W. Ellis, *"Robust Landmark-Based Audio Fingerprinting,"* LabROSA, Columbia University, 2009. <https://www.ee.columbia.edu/~dpwe/LabROSA/matlab/fingerprint/>

---

## Notes

This project was developed for the **EE200** course. The fingerprinting engine (`fingerprint.py`) is self-contained and independent of the Streamlit front end, so it can also be imported and used programmatically.
