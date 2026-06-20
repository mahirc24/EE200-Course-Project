"""
fingerprint.py
==============
A compact, Shazam-style audio identifier (constellation peaks + combinatorial
hashing + offset-histogram voting).

Pipeline (Wang 2003; implementation style after Ellis' landmark fingerprinter):
  audio  ->  spectrogram  ->  peak constellation  ->  paired hashes
  query hashes  ->  matched DB hashes  ->  per-song offset histogram  ->  winner

References
----------
[1] A. L.-C. Wang, "An Industrial-Strength Audio Search Algorithm," ISMIR 2003.
[2] D. P. W. Ellis, "Robust Landmark-Based Audio Fingerprinting," LabROSA,
    Columbia University, 2009.  https://www.ee.columbia.edu/~dpwe/LabROSA/matlab/fingerprint/
"""
from __future__ import annotations
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter

# ----------------------------------------------------------------------------- config
SR        = 11025      # all audio is resampled to this rate (mono)
N_FFT     = 1024       # STFT window length (samples)
HOP       = 256        # STFT hop (samples)  -> ~23 ms frames
# peak picking
PEAK_NEIGH_T = 19      # local-max neighbourhood in time frames
PEAK_NEIGH_F = 19      # local-max neighbourhood in frequency bins
PEAK_MIN_DB  = -55.0   # ignore peaks quieter than this (relative to max = 0 dB)
# combinatorial hashing (anchor -> target zone)
FAN_OUT   = 8          # max points paired with each anchor
DT_MIN    = 1          # min time gap (frames) between anchor and target
DT_MAX    = 40         # max time gap (frames)
DF_MAX    = 80         # max |freq-bin| difference inside the target zone


# ----------------------------------------------------------------------------- spectrogram
def compute_spectrogram(y, sr=SR, n_fft=N_FFT, hop=HOP):
    """Return (freqs, times, S_db): magnitude spectrogram in dB (max = 0 dB)."""
    f, t, Z = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    S = np.abs(Z)
    S_db = 20.0 * np.log10(S + 1e-10)
    S_db -= S_db.max()                      # normalise so the loudest bin is 0 dB
    return f, t, S_db


# ----------------------------------------------------------------------------- constellation
def find_peaks(S_db, neigh_t=PEAK_NEIGH_T, neigh_f=PEAK_NEIGH_F, min_db=PEAK_MIN_DB):
    """Return peak list as an (n,2) int array of [freq_bin, time_frame] coordinates.

    A bin is a peak if it is the maximum in a (neigh_f x neigh_t) window and is
    louder than `min_db`.  Amplitude is then discarded (only coordinates remain).
    """
    footprint = np.ones((neigh_f, neigh_t), dtype=bool)
    local_max = (S_db == maximum_filter(S_db, footprint=footprint))
    peaks_mask = local_max & (S_db > min_db)
    fb, tf = np.where(peaks_mask)
    return np.stack([fb, tf], axis=1)


# ----------------------------------------------------------------------------- hashing
def _pack(f1, f2, dt):
    """Pack a hash into a 32-bit int: 10 bits f1 | 10 bits f2 | 10 bits dt (Wang 2003)."""
    return ((int(f1) & 0x3FF) << 20) | ((int(f2) & 0x3FF) << 10) | (int(dt) & 0x3FF)


def make_hashes(peaks, fan_out=FAN_OUT, dt_min=DT_MIN, dt_max=DT_MAX, df_max=DF_MAX):
    """Combinatorial hashing: pair each anchor peak with peaks in its target zone.

    Returns a list of (hash_int, anchor_time_frame) tuples.
    """
    if len(peaks) == 0:
        return []
    pk = peaks[np.argsort(peaks[:, 1])]        # sort by time frame
    f = pk[:, 0]; t = pk[:, 1]
    hashes = []
    n = len(pk)
    for i in range(n):
        f1, t1 = f[i], t[i]
        paired = 0
        j = i + 1
        while j < n and paired < fan_out:
            dt = t[j] - t1
            if dt < dt_min:
                j += 1; continue
            if dt > dt_max:
                break                          # peaks are time-sorted: no farther targets
            if abs(int(f[j]) - int(f1)) <= df_max:
                hashes.append((_pack(f1, f[j], dt), int(t1)))
                paired += 1
            j += 1
    return hashes


def single_peak_tokens(peaks):
    """Degenerate 'fingerprint' using individual peaks (frequency only) as tokens.

    Used only to contrast single-peak matching against paired hashing.
    Returns a list of (freq_token, time_frame) tuples.
    """
    return [(int(fb), int(tf)) for fb, tf in peaks]


# ----------------------------------------------------------------------------- fingerprint one clip
def fingerprint(y, sr=SR, **kw):
    """audio -> (peaks, hashes).  Convenience wrapper around the steps above."""
    _, _, S_db = compute_spectrogram(y, sr=sr)
    peaks = find_peaks(S_db)
    hashes = make_hashes(peaks, **kw)
    return peaks, hashes


# ----------------------------------------------------------------------------- database
class FingerprintDB:
    """Inverted index: hash -> list of (song_id, anchor_time_frame)."""

    def __init__(self):
        self.index: dict[int, list[tuple[int, int]]] = {}
        self.songs: list[str] = []             # song_id -> label (filename w/o ext)

    def add_song(self, label, y, sr=SR, **kw):
        sid = len(self.songs)
        self.songs.append(label)
        _, hashes = fingerprint(y, sr=sr, **kw)
        for h, t in hashes:
            self.index.setdefault(h, []).append((sid, t))
        return sid, len(hashes)

    # ---- matching ----
    def match(self, y, sr=SR, **kw):
        """Identify a query clip.

        Returns (best_label, score, results) where `results` is a sorted list of
        (label, score, best_offset) and the per-song offset histograms are used
        for scoring.  Also returns the raw match pairs for plotting.
        """
        _, q_hashes = fingerprint(y, sr=sr, **kw)
        # collect offset votes per song
        votes: dict[int, dict[int, int]] = {}      # song_id -> {offset: count}
        scatter: dict[int, list[tuple[int, int]]] = {}  # song_id -> [(db_t, q_t)]
        for h, q_t in q_hashes:
            for sid, db_t in self.index.get(h, ()):
                off = db_t - q_t
                votes.setdefault(sid, {}).setdefault(off, 0)
                votes[sid][off] += 1
                scatter.setdefault(sid, []).append((db_t, q_t))

        results = []
        for sid, hist in votes.items():
            best_off = max(hist, key=hist.get)
            results.append((self.songs[sid], hist[best_off], best_off))
        results.sort(key=lambda r: r[1], reverse=True)

        if not results:
            return None, 0, [], scatter, votes
        best_label, best_score, _ = results[0]
        return best_label, best_score, results, scatter, votes

    def match_single_peaks(self, y, sr=SR):
        """Same voting scheme but using single-peak tokens (for comparison)."""
        _, _, S_db = compute_spectrogram(y, sr=sr)
        q_tokens = single_peak_tokens(find_peaks(S_db))
        # build a single-peak index on the fly from stored songs is expensive;
        # instead this is only used in the demo where a dedicated index is built.
        raise NotImplementedError("Use build_single_peak_index() in the demo.")


# ----------------------------------------------------------------------------- I/O helper
def load_audio(path, sr=SR):
    """Load any audio file as mono float32 at `sr` (needs librosa + ffmpeg)."""
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)
