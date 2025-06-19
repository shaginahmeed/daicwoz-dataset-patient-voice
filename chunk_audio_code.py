
import pandas as pd
from pydub import AudioSegment
import webrtcvad
import numpy as np
import noisereduce as nr
import os
import shutil

source_base_dir = r"D:\FYDP\FYDP DATASET"
output_base_dir = r"E:\FYDP DATASET\daicwoz-dataset-patient-voice"

vad_aggressiveness = 3 
frame_duration_ms = 30
buffer_ms = 10
merge_threshold_ms = 5000

def time_to_ms(t):
    try:
        return int(float(t) * 1000)
    except:
        return None

def process_folder(folder_name):
    print(f"\n[INFO] Processing folder: {folder_name}")

    folder_path = os.path.join(source_base_dir, folder_name)
    folder_prefix = folder_name[:-2] if folder_name.endswith('_P') else folder_name

    audio_path = os.path.join(folder_path, f"{folder_prefix}_AUDIO.wav")
    transcript_path = os.path.join(folder_path, f"{folder_prefix}_TRANSCRIPT.csv")

    if not os.path.isfile(audio_path) or not os.path.isfile(transcript_path):
        print(f"[WARN] Required files missing in {folder_path}")
        return

    out_folder = os.path.join(output_base_dir, folder_prefix)
    os.makedirs(out_folder, exist_ok=True)
    output_transcript_path = os.path.join(out_folder, f"{folder_prefix}_TRANSCRIPT.csv")

    df_raw = pd.read_csv(transcript_path, sep="\t")
    df_raw.columns = df_raw.columns.str.strip().str.lower()

    speaker_counts = df_raw['speaker'].value_counts()
    patient_speakers = [s for s in speaker_counts.index if str(s).strip().lower() != 'ellie']
    if not patient_speakers:
        print("[ERROR] No patient speaker found.")
        return

    patient_speaker = patient_speakers[0]
    print(f"[INFO] Using '{patient_speaker}' as patient speaker.")

    def is_clean_patient_voice(row):
        speaker = str(row['speaker']).strip().lower()
        value = str(row['value']).strip().lower()
        if speaker != str(patient_speaker).strip().lower():
            return False
        if not value or any(tag in value for tag in ['[silence]', '[noise]', '[laughter]']):
            return False
        return True

    df_raw['start_ms'] = df_raw['start_time'].apply(time_to_ms)
    df_raw['end_ms'] = df_raw['stop_time'].apply(time_to_ms)
    df = df_raw[df_raw.apply(is_clean_patient_voice, axis=1)].copy()

    if df.empty:
        print("[WARN] No clean patient segments found.")
        return

    original_audio = AudioSegment.from_wav(audio_path)
    audio = original_audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    sample_rate = 16000
    sample_width = 2
    bytes_per_frame = int(sample_rate * frame_duration_ms / 1000) * sample_width
    vad = webrtcvad.Vad(vad_aggressiveness)

    def vad_split_segment(segment, global_start_ms):
        raw = segment.raw_data
        speech_ranges = []
        in_speech = False
        temp_start = None

        for i in range(0, len(raw) - bytes_per_frame + 1, bytes_per_frame):
            frame = raw[i:i + bytes_per_frame]
            is_speech = vad.is_speech(frame, sample_rate)
            if is_speech and not in_speech:
                in_speech = True
                temp_start = i
            elif not is_speech and in_speech:
                in_speech = False
                speech_ranges.append((temp_start, i))

        if in_speech:
            speech_ranges.append((temp_start, len(raw)))

        merged_ranges = []
        for start, end in speech_ranges:
            start_ms = global_start_ms + int((start / sample_width) * 1000 / sample_rate) - buffer_ms
            end_ms = global_start_ms + int((end / sample_width) * 1000 / sample_rate) + buffer_ms
            start_ms = max(0, start_ms)
            end_ms = min(len(original_audio), end_ms)
            merged_ranges.append((start_ms, end_ms))
        return merged_ranges

    full_patient_audio = AudioSegment.empty()
    for _, row in df.iterrows():
        start, end = row['start_ms'], row['end_ms']
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        segment = audio[start:end]
        speech_times = vad_split_segment(segment, start)
        for s_start, s_end in speech_times:
            clip = original_audio[s_start:s_end]
            samples = np.array(clip.get_array_of_samples()).astype(np.float32)
            reduced = nr.reduce_noise(y=samples, sr=16000)
            clean_clip = AudioSegment(
                reduced.astype(np.int16).tobytes(),
                frame_rate=16000,
                sample_width=2,
                channels=1
            )
            full_patient_audio += clean_clip

    full_out_path = os.path.join(out_folder, f"PATIENT_AUDIO_{folder_prefix}.wav")
    full_patient_audio.export(full_out_path, format="wav")
    print(f"[✓] Exported full patient audio: {full_out_path}")

    chunk_count = 0
    chunk_log = []
    pending_chunk = AudioSegment.empty()
    pending_start = None
    pending_text = []
    pending_indexes = []

    def ms_to_minsec_ms(ms):
        minutes = ms // 60000
        seconds = (ms % 60000) // 1000
        millis = ms % 1000
        return f"{minutes:02d}:{seconds:02d}.{millis:03d}"

    for idx, row in df.iterrows():
        start, end = row['start_ms'], row['end_ms']
        value = str(row['value']).strip()
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue

        clip = original_audio[start:end]
        samples = np.array(clip.get_array_of_samples()).astype(np.float32)
        reduced = nr.reduce_noise(y=samples, sr=16000)
        clean_clip = AudioSegment(
            reduced.astype(np.int16).tobytes(),
            frame_rate=16000,
            sample_width=2,
            channels=1
        )

        if pending_chunk.duration_seconds * 1000 < merge_threshold_ms:
            if pending_start is None:
                pending_start = start
            pending_chunk += clean_clip
            pending_text.append(value)
            pending_indexes.append(idx)
            continue
        else:
            chunk_count += 1
            chunk_filename = f"{folder_prefix}_{chunk_count}_PATIENT_AUDIO.wav"
            chunk_path = os.path.join(out_folder, chunk_filename)
            pending_chunk.export(chunk_path, format="wav")

            chunk_log.append({
                'Chunk Name': chunk_filename,
                'Start': ms_to_minsec_ms(pending_start),
                'End': ms_to_minsec_ms(start),
                'Duration (s)': round(pending_chunk.duration_seconds, 2),
                'Utterance': ' '.join(pending_text),
                'Transcript Indexes': '"' + ','.join(map(str, pending_indexes)) + '"',
                'Merged Count': len(pending_text)
})

            pending_chunk = clean_clip
            pending_start = start
            pending_text = [value]
            pending_indexes = [idx]

    if len(pending_chunk) > 0:
        chunk_count += 1
        chunk_filename = f"{folder_prefix}_{chunk_count}_PATIENT_AUDIO.wav"
        chunk_path = os.path.join(out_folder, chunk_filename)
        pending_chunk.export(chunk_path, format="wav")

        chunk_log.append({
            'Chunk Name': chunk_filename,
            'Start': ms_to_minsec_ms(pending_start),
            'End': ms_to_minsec_ms(end),
            'Duration (s)': round(pending_chunk.duration_seconds, 2),
            'Utterance': ' '.join(pending_text),
            'Transcript Indexes': '"' + ','.join(map(str, pending_indexes)) + '"',
            'Merged Count': len(pending_text)
})

    if chunk_log:
        shutil.copy2(transcript_path, output_transcript_path)
        df_log = pd.DataFrame(chunk_log)
        log_csv_path = os.path.join(out_folder, f"{folder_prefix}_chunk_metadata.csv")
        df_log.to_csv(log_csv_path, index=False)
        print(f"[✓] Saved chunk metadata CSV: {log_csv_path}")
    else:
        print("[!] ERROR: No chunks were saved.")

def main():
    folders = [f for f in os.listdir(source_base_dir)
               if os.path.isdir(os.path.join(source_base_dir, f)) and f.endswith('_P')]
    print(f"[INFO] Found {len(folders)} folders to process.")
    for folder in folders:
        process_folder(folder)

if __name__ == "__main__":
    main()