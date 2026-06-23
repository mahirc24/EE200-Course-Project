"""
app.py - "Zapp-tain America": a Shazam-style song identifier (Streamlit).

Two modes:
  * Single clip - identify one uploaded clip and show the intermediate steps
    (spectrogram, constellation of peaks, offset histogram that decides the match).
  * Batch - identify a set of clips and download results.csv with columns
    exactly: filename,prediction.
"""

import os
import tempfile
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import fingerprint as fp

# ─────────────────────────────────────────────────
# Page config  (must be FIRST streamlit call)
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Zapp-tain America",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────
# Inject custom CSS from styles.css
# ─────────────────────────────────────────────────
_CSS_PATH = pathlib.Path(__file__).parent / "styles.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp { background-color: #0D1117; color: #E6EDF3; }
    .stButton > button { background: #7C3AED !important; color: #fff !important;
        border: none !important; border-radius: 10px !important; }
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# Matplotlib dark theme
# ─────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#1C2333",
    "axes.facecolor":    "#1C2333",
    "axes.edgecolor":    "#30363D",
    "axes.labelcolor":   "#8B949E",
    "xtick.color":       "#8B949E",
    "ytick.color":       "#8B949E",
    "text.color":        "#E6EDF3",
    "grid.color":        "#30363D",
    "grid.alpha":        0.4,
    "figure.dpi":        120,
})

DB_PATH      = "database.pkl"
MAX_FREQ_HZ  = 4000

# ─────────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────────
def _load_pickle(path):
    import gzip, pickle
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):
        with open(path, "rb") as f:
            return pickle.load(f)


@st.cache_resource
def load_db(path=DB_PATH):
    if not os.path.exists(path):
        return None
    data = _load_pickle(path)
    db = fp.FingerprintDB()
    db.index = data["index"]
    db.songs  = data["songs"]
    return db


def read_upload(uploaded, max_seconds=fp.MAX_QUERY_SECONDS):
    suffix = os.path.splitext(uploaded.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        path = tmp.name
    try:
        y = fp.load_audio(path, duration=max_seconds)
    finally:
        os.unlink(path)
    return y

# ─────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────
def plot_spectrogram(f, t, S):
    fmask = f <= MAX_FREQ_HZ
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(
        S[fmask], origin="lower", aspect="auto",
        extent=[float(t[0]), float(t[-1]), 0.0, MAX_FREQ_HZ],
        cmap="magma", vmin=-60, vmax=0,
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Spectrogram", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_constellation(f, t, peaks):
    fig, ax = plt.subplots(figsize=(6, 3))
    if len(peaks):
        ax.scatter(t[peaks[:, 1]], f[peaks[:, 0]], s=6, c="#7C3AED", alpha=0.8)
    ax.set_ylim(0, MAX_FREQ_HZ)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(f"Constellation  ({len(peaks)} peaks)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_histogram(votes, db, best_label):
    fig, ax = plt.subplots(figsize=(6, 3))
    if votes and best_label in db.songs:
        sid = db.songs.index(best_label)
        h   = votes.get(sid, {})
        if h:
            offs = np.array(list(h.keys())) * fp.HOP / fp.SR
            ax.bar(offs, list(h.values()), width=0.05, color="#14B8A6", alpha=0.9)
    ax.set_xlabel("offset  t_db - t_query (s)")
    ax.set_ylabel("aligned hashes")
    ax.set_title(f"Best match: {best_label}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def show_fig(col, fig):
    col.pyplot(fig)
    plt.close(fig)

# ─────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────
st.markdown("""
<div style="
    display:flex; align-items:center; gap:14px;
    padding:0.5rem 0 1.5rem;
    border-bottom:1px solid #30363D;
    margin-bottom:1.5rem;
">
    <span style="font-size:2.6rem;">🎵</span>
    <div>
        <h1 style="margin:0;font-size:1.9rem;font-weight:700;
                   color:#E6EDF3;letter-spacing:-0.5px;">
            Zapp-tain America
        </h1>
        <p style="margin:2px 0 0;font-size:0.82rem;color:#8B949E;">
            Audio fingerprinting &nbsp;·&nbsp; constellation peaks &nbsp;·&nbsp;
            offset-histogram voting (Wang 2003)
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# Load database
# ─────────────────────────────────────────────────
db = load_db()
if db is None:
    st.error("**database.pkl not found.** Build the song library into `database.pkl` first.")
    st.stop()

# Stat cards
col_stat1, col_stat2, col_stat3 = st.columns(3)
for col, label, value, color in [
    (col_stat1, "Songs indexed",   str(len(db.songs)),       "#14B8A6"),
    (col_stat2, "Distinct hashes", f"{len(db.index):,}",     "#7C3AED"),
    (col_stat3, "Max query",       f"{fp.MAX_QUERY_SECONDS}s", "#F59E0B"),
]:
    col.markdown(f"""
<div style="
    background:#1C2333;border:1px solid #30363D;border-radius:10px;
    padding:0.9rem 1.1rem;text-align:center;
">
    <p style="margin:0;font-size:0.72rem;color:#8B949E;
              text-transform:uppercase;letter-spacing:1px;">{label}</p>
    <p style="margin:4px 0 0;font-size:1.8rem;font-weight:700;color:{color};">{value}</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

with st.expander("📋  Songs in the database"):
    st.markdown(
        "<p style='color:#8B949E;font-size:0.85rem;line-height:1.9;'>"
        + " &nbsp;·&nbsp; ".join(db.songs)
        + "</p>",
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# Controls row
# ─────────────────────────────────────────────────
ctrl_left, ctrl_right = st.columns([1, 2])
with ctrl_left:
    st.markdown(
        "<p style='font-size:0.78rem;color:#8B949E;margin-bottom:4px;"
        "text-transform:uppercase;letter-spacing:0.8px;'>Mode</p>",
        unsafe_allow_html=True,
    )
    mode = st.radio("Mode", ["Single clip", "Batch"], horizontal=True, label_visibility="collapsed")
with ctrl_right:
    threshold = st.slider(
        "Match threshold  (minimum aligned hashes)",
        min_value=1, max_value=50, value=10,
        help=(
            "Genuine matches score hundreds-thousands of aligned hashes. "
            "Clips not in the database score only a handful."
        ),
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ╔═════════════════════════════════════════╗
# ║           SINGLE CLIP MODE             ║
# ╚═════════════════════════════════════════╝
if mode == "Single clip":
    st.markdown("<h3 style='margin-bottom:0.4rem;'>🎧  Upload a query clip</h3>",
                unsafe_allow_html=True)
    st.caption("Supported: WAV · MP3 · M4A · FLAC · OGG")

    up = st.file_uploader(
        "Drop or browse",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        label_visibility="collapsed",
    )

    if "single_file_name" not in st.session_state:
        st.session_state.single_file_name = None
        st.session_state.single_result    = None

    if up is not None:
        if st.session_state.single_file_name != up.name:
            st.session_state.single_file_name = up.name
            st.session_state.single_result    = None

        st.audio(up)
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

        st.markdown("<div class='pulse-btn'>", unsafe_allow_html=True)
        run = st.button("🎵  Identify Song", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

        if run:
            with st.spinner("Analysing audio fingerprint…"):
                try:
                    y = read_upload(up)
                    f, t, S = fp.compute_spectrogram(y)
                    peaks   = fp.find_peaks(S)
                    label, score, results, votes = db.match_hashes(fp.make_hashes(peaks))
                    st.session_state.single_result = dict(
                        f=f, t=t, S=S, peaks=peaks,
                        label=label, score=score,
                        results=results, votes=votes,
                    )
                except Exception as e:
                    st.error(f"Could not read this file: {e}")

        res = st.session_state.single_result
        if res is not None:
            f, t, S = res["f"], res["t"], res["S"]
            peaks   = res["peaks"]
            label, score, results, votes = (
                res["label"], res["score"], res["results"], res["votes"]
            )

            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            show_fig(c1, plot_spectrogram(f, t, S))
            show_fig(c2, plot_constellation(f, t, peaks))
            if label is not None:
                show_fig(c3, plot_histogram(votes, db, label))

            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

            if label is None or score < threshold:
                best_str = (
                    f" — best guess **{label}** scored only **{score}**"
                    if label else ""
                )
                st.error(f"Not in database{best_str}. "
                         f"No song cleared the threshold of {threshold} aligned hashes.")
            else:
                st.markdown(f"""
<div style="
    background:rgba(20,184,166,0.10);
    border:1px solid #14B8A6;
    border-radius:12px;
    padding:1.2rem 1.5rem;
    display:flex;align-items:center;gap:16px;
">
    <span style="font-size:2rem;">✅</span>
    <div>
        <p style="margin:0;font-size:0.75rem;color:#14B8A6;
                  text-transform:uppercase;letter-spacing:1px;">Identified</p>
        <p style="margin:4px 0 0;font-size:1.4rem;font-weight:700;color:#E6EDF3;">{label}</p>
        <p style="margin:2px 0 0;font-size:0.82rem;color:#8B949E;">
            Score: <strong style="color:#14B8A6;">{score}</strong> aligned hashes
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

            if results:
                st.markdown("<h4 style='margin:1.2rem 0 0.4rem;'>Top candidates</h4>",
                            unsafe_allow_html=True)
                st.dataframe(
                    pd.DataFrame(results,
                                 columns=["song", "score", "best_offset(frames)"]).head(5),
                    hide_index=True, use_container_width=True,
                )

# ╔═════════════════════════════════════════╗
# ║              BATCH MODE                ║
# ╚═════════════════════════════════════════╝
else:
    st.markdown("<h3 style='margin-bottom:0.4rem;'>📦  Batch identification</h3>",
                unsafe_allow_html=True)
    st.caption("Upload multiple clips — results download as results.csv")

    ups = st.file_uploader(
        "Drop or browse files",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if "batch_file_names" not in st.session_state:
        st.session_state.batch_file_names = []
        st.session_state.batch_result_df  = None

    if ups:
        current_names = [u.name for u in ups]
        if current_names != st.session_state.batch_file_names:
            st.session_state.batch_file_names = current_names
            st.session_state.batch_result_df  = None

        preview = ", ".join(u.name for u in ups[:6]) + ("…" if len(ups) > 6 else "")
        st.markdown(f"""
<div style="
    background:#1C2333;border:1px solid #30363D;border-radius:10px;
    padding:0.9rem 1.1rem;margin:0.6rem 0 1rem;
">
    <p style="margin:0;font-size:0.78rem;color:#8B949E;">
        <strong style="color:#E6EDF3;">{len(ups)}</strong> file(s) ready &nbsp;·&nbsp; {preview}
    </p>
</div>
""", unsafe_allow_html=True)

        run_batch = st.button("🚀  Run Batch", type="primary")

        if run_batch:
            rows = []
            prog             = st.progress(0.0)
            status_slot      = st.empty()
            for i, up in enumerate(ups):
                status_slot.caption(f"Processing **{up.name}**  ({i+1}/{len(ups)})…")
                try:
                    y = read_upload(up)
                    label, score, *_ = db.match(y)
                    pred = label if (label and score >= threshold) else "not_in_database"
                except Exception as e:
                    pred = f"error: {e}"
                rows.append({"filename": up.name, "prediction": pred})
                prog.progress((i + 1) / len(ups))
            status_slot.empty()
            st.session_state.batch_result_df = pd.DataFrame(
                rows, columns=["filename", "prediction"]
            )

        df = st.session_state.batch_result_df
        if df is not None:
            st.markdown("<h4 style='margin:1rem 0 0.4rem;'>Results</h4>",
                        unsafe_allow_html=True)
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.download_button(
                "⬇  Download results.csv",
                data=df.to_csv(index=False).encode(),
                file_name="results.csv",
                mime="text/csv",
            )

# ─────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────
st.markdown("<hr style='margin-top:3rem;'>", unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center;font-size:0.78rem;color:#484F58;padding-bottom:1rem;">
    Zapp-tain America &nbsp;·&nbsp; EE200 Course Project &nbsp;·&nbsp;
    <a href="https://github.com/MadhusudanKantharia/EE200-Course-Project"
       style="color:#7C3AED;text-decoration:none;">GitHub ↗</a>
</p>
""", unsafe_allow_html=True)
