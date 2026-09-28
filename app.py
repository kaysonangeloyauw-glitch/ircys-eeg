import streamlit as st
import numpy as np
import pandas as pd
import scipy.signal as signal
import plotly.express as px
from PIL import Image
import librosa
import mne
import tempfile

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EEG Frequency Priority Advisor",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 EEG Frequency Band Priority Recommendation System")
st.markdown("""
This application analyzes uploaded EEG time-series data or spectrogram images, computes 
**relative band power** across standard frequency bands (1–45 Hz), and generates 
actionable priority recommendations for frequency adjustments.
""")

# --- CONFIGURATION & BANDS ---
L_FREQ = 1.0
H_FREQ = 45.0

BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta":  (13.0, 30.0),
    "Gamma": (30.0, 45.0)
}

# --- SIDEBAR: CLINICAL / TARGET GOAL ---
st.sidebar.header("1. Target Goal Selection")
target_goal = st.sidebar.selectbox(
    "Select Target Mental State / Objective:",
    [
        "Deep Focus & Cognition (Boost Beta/Gamma)",
        "Stress Reduction & Calm (Boost Alpha)",
        "Deep Sleep Optimization (Boost Delta)",
        "Drowsiness Reduction (Reduce Theta/Alpha)"
    ]
)

# --- CORE SIGNAL PROCESSING FUNCTIONS ---
def preprocess_and_compute_psd(data, fs):
    """Applies 1-45Hz bandpass filter and calculates Welch's PSD & Relative Band Power."""
    nyquist = 0.5 * fs
    low = L_FREQ / nyquist
    high = min(H_FREQ / nyquist, 0.99)
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)

    nperseg = min(len(filtered_data), int(fs * 2))
    freqs, psd = signal.welch(filtered_data, fs=fs, nperseg=nperseg, detrend="constant")

    total_mask = (freqs >= L_FREQ) & (freqs <= H_FREQ)
    total_power = np.trapz(psd[total_mask], freqs[total_mask]) + 1e-20

    rel_powers = {}
    for band_name, (low_f, high_f) in BANDS.items():
        band_mask = (freqs >= low_f) & (freqs <= high_f)
        band_power = np.trapz(psd[band_mask], freqs[band_mask])
        rel_powers[band_name] = float((band_power / total_power) * 100.0)

    return freqs, psd, rel_powers

def process_spectrogram_image(pil_image):
    """Estimates relative band power by mapping image y-axis pixel intensity to frequency bands."""
    gray_img = pil_image.convert('L')
    img_arr = np.array(gray_img, dtype=float)
    
    if np.mean(img_arr) > 127:
        img_arr = 255.0 - img_arr

    freq_profile = np.mean(img_arr, axis=1)[::-1]
    num_rows = len(freq_profile)
    freq_axis = np.linspace(L_FREQ, H_FREQ, num_rows)
    
    total_intensity = np.sum(freq_profile) + 1e-20
    rel_powers = {}

    for band_name, (low_f, high_f) in BANDS.items():
        mask = (freq_axis >= low_f) & (freq_axis <= high_f)
        band_intensity = np.sum(freq_profile[mask])
        rel_powers[band_name] = float((band_intensity / total_intensity) * 100.0)

    return rel_powers

def generate_priority_recommendations(rel_powers, goal):
    """Evaluates relative band powers against target goals to generate prioritized band shift advice."""
    recs = []
    
    delta = rel_powers.get("Delta", 0)
    theta = rel_powers.get("Theta", 0)
    alpha = rel_powers.get("Alpha", 0)
    beta  = rel_powers.get("Beta", 0)
    gamma = rel_powers.get("Gamma", 0)

    if "Deep Focus" in goal:
        if beta < 25.0:
            recs.append(("High Priority", "INCREASE **Beta (13–30 Hz)**", f"Current Beta power is low ({beta:.1f}%). Beta enhancement is recommended to support active cognitive processing."))
        if theta > 25.0:
            recs.append(("Medium Priority", "DECREASE **Theta (4–8 Hz)**", f"Elevated Theta power ({theta:.1f}%) during cognitive tasks can indicate drowsiness or attention drift."))
        if gamma < 5.0:
            recs.append(("Low Priority", "INCREASE **Gamma (30–45 Hz)**", f"Gamma power ({gamma:.1f}%) is low; subtle increases may improve complex information integration."))

    elif "Stress Reduction" in goal:
        if alpha < 30.0:
            recs.append(("High Priority", "INCREASE **Alpha (8–13 Hz)**", f"Current Alpha power ({alpha:.1f}%) is below ideal relaxation baseline. Prioritize increasing sensorimotor/occipital Alpha."))
        if beta > 35.0:
            recs.append(("High Priority", "DECREASE **Beta (13–30 Hz)**", f"High Beta activity ({beta:.1f}%) is associated with cognitive strain and anxiety."))

    elif "Deep Sleep" in goal:
        if delta < 45.0:
            recs.append(("High Priority", "INCREASE **Delta (1–4 Hz)**", f"Delta power ({delta:.1f}%) is critical for restorative slow-wave sleep."))
        if beta > 15.0:
            recs.append(("Medium Priority", "DECREASE **Beta (13–30 Hz)**", f"High Beta ({beta:.1f}%) inhibits sleep onset and deep sleep transitions."))

    elif "Drowsiness Reduction" in goal:
        if theta > 30.0 or alpha > 35.0:
            recs.append(("High Priority", "DECREASE **Theta/Alpha (4–13 Hz)**", f"Elevated low-frequency power (Theta: {theta:.1f}%, Alpha: {alpha:.1f}%) correlates with reduced vigilance."))
        if beta < 20.0:
            recs.append(("Medium Priority", "INCREASE **Beta (13–30 Hz)**", f"Increase Beta power ({beta:.1f}%) to promote alert wakefulness."))

    if not recs:
        recs.append(("Optimal", "MAINTAIN Current Spectrum", "Your current EEG frequency distribution closely aligns with the selected target state."))

    return recs

def display_results(rel_powers, goal):
    """Renders charts, metrics, and priority advice."""
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Relative Frequency Band Power (%)")
        fig_bar = px.bar(
            x=list(rel_powers.keys()),
            y=list(rel_powers.values()),
            labels={'x': 'Band', 'y': 'Relative Power (%)'},
            color=list(rel_powers.keys()),
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("### Summary Metrics")
        for band, val in rel_powers.items():
            st.metric(label=f"{band} Band Power", value=f"{val:.2f} %")

    st.markdown("---")
    st.subheader("3. Recommended Frequency Band Adjustment Priorities")
    recommendations = generate_priority_recommendations(rel_powers, goal)

    for priority, title, detail in recommendations:
        if priority == "High Priority":
            st.error(f"🔴 **[{priority}]** {title}\n\n{detail}")
        elif priority == "Medium Priority":
            st.warning(f"🟡 **[{priority}]** {title}\n\n{detail}")
        elif priority == "Low Priority":
            st.info(f"🔵 **[{priority}]** {title}\n\n{detail}")
        else:
            st.success(f"🟢 **[{priority}]** {title}\n\n{detail}")

# --- UI INPUT SECTION ---
st.subheader("2. Upload EEG File or Image")
uploaded_files = st.file_uploader(
    "Upload EEG Signal (.edf, .bdf, .csv, .wav) or Spectrogram Image (.png, .jpg, .jpeg). Multiple files will be aggregated using median values.",
    type=["edf", "bdf", "csv", "wav", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    file_type = uploaded_files[0].name.split('.')[-1].lower()

    # SCENARIO A: OpenNeuro Standard Files (.edf / .bdf)
    if file_type in ['edf', 'bdf']:
        all_rel_powers = []
        for uploaded_file in uploaded_files:
            try:
                # Write file temporarily so MNE can read it directly from disk
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name

                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False) if file_type == 'edf' else mne.io.read_raw_bdf(tmp_path, preload=True, verbose=False)
                fs = int(raw.info['sfreq'])
                
                # Pick EEG channels and extract signal data from first available channel
                raw.pick_types(eeg=True)
                data = raw.get_data()[0]  # First EEG channel
                
                freqs, psd, rel_powers = preprocess_and_compute_psd(data, fs)
                all_rel_powers.append(rel_powers)
            except Exception as e:
                st.error(f"Error reading {uploaded_file.name}: {e}")

        if all_rel_powers:
            aggregated_powers = {
                b: float(np.median([p[b] for p in all_rel_powers])) for b in BANDS.keys()
            }
            display_results(aggregated_powers, target_goal)

    # SCENARIO B: CSV Time-Series Input
    elif file_type == 'csv':
        all_rel_powers = []
        fs = st.sidebar.number_input("Sampling Frequency (Hz):", min_value=1, max_value=2048, value=256)

        for uploaded_file in uploaded_files:
            try:
                df = pd.read_csv(uploaded_file)
                signal_data = df.select_dtypes(include=[np.number]).iloc[:, 0].to_numpy()
                freqs, psd, rel_powers = preprocess_and_compute_psd(signal_data, fs)
                all_rel_powers.append(rel_powers)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

        if all_rel_powers:
            aggregated_powers = {
                b: float(np.median([p[b] for p in all_rel_powers])) for b in BANDS.keys()
            }
            display_results(aggregated_powers, target_goal)

    # SCENARIO C: Audio Wave Input (.wav)
    elif file_type == 'wav':
        audio_file = uploaded_files[0]
        data, fs = librosa.load(audio_file, sr=None)
        freqs, psd, rel_powers = preprocess_and_compute_psd(data, fs)

        st.audio(audio_file)
        display_results(rel_powers, target_goal)

    # SCENARIO D: Spectrogram Image Input (.png, .jpg, .jpeg)
    elif file_type in ['png', 'jpg', 'jpeg']:
        all_rel_powers = []
        for img_file in uploaded_files:
            image = Image.open(img_file)
            st.image(image, caption=f"Uploaded Spectrogram: {img_file.name}", use_container_width=True)
            rel_powers = process_spectrogram_image(image)
            all_rel_powers.append(rel_powers)

        if all_rel_powers:
            aggregated_powers = {
                b: float(np.median([p[b] for p in all_rel_powers])) for b in BANDS.keys()
            }
            display_results(aggregated_powers, target_goal)