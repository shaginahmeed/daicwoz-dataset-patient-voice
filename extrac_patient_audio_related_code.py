
# This script extracts patient audio files from zip archives in a specified directory.

# import zipfile
# import os

# base_dir = r"D:\FYDP\FYDP DATASET"

# for file in os.listdir(base_dir):
#     if file.endswith("_P.zip"):
#         zip_path = os.path.join(base_dir, file)
#         folder_name = file.replace(".zip", "")
#         extract_path = os.path.join(base_dir, folder_name)

#         if not os.path.exists(extract_path):
#             print(f"Extracting: {file}")
#             with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#                 zip_ref.extractall(extract_path)
#         else:
#             print(f"Already extracted: {folder_name}")











# This script processes audio files and transcripts to extract segments for participants.

# import os
# import pandas as pd
# import soundfile as sf
# import numpy as np

# BASE_DIR = r"D:\FYDP\FYDP DATASET"

# for pid in range(300, 403):
#     folder = os.path.join(BASE_DIR, f"{pid}_P")
#     wav_path = os.path.join(folder, f"{pid}_AUDIO.wav")
#     txt_path = os.path.join(folder, f"{pid}_TRANSCRIPT.csv")
#     out_path = os.path.join(folder, f"{pid}_patient.wav")

#     if not os.path.exists(wav_path) or not os.path.exists(txt_path):
#         print(f"Missing files for: {pid}_P")
#         continue

#     try:
#         # Read CSV with tab delimiter
#         df = pd.read_csv(txt_path, sep='\t')

#         # Check for 'speaker' column
#         if 'speaker' not in df.columns:
#             print(f"Error for {pid}_P: 'speaker' column missing. Found columns: {list(df.columns)}")
#             continue

#         audio, sr = sf.read(wav_path)

#         # Select only participant segments
#         segments = df[df['speaker'] == 'Participant']

#         if segments.empty:
#             print(f"No 'Participant' segments for {pid}_P")
#             continue

#         # Concatenate participant audio segments
#         patient_audio = np.concatenate([
#             audio[int(float(row['start_time']) * sr): int(float(row['stop_time']) * sr)]
#             for _, row in segments.iterrows()
#         ])

#         # Save the patient-only audio file
#         sf.write(out_path, patient_audio, sr)
#         print(f"Saved: {out_path}")

#     except Exception as e:
#         print(f"Error for {pid}_P: {e}")










# This script denoises patient audio files using the noisereduce library.

#not working, need to fix
# import os
# import soundfile as sf
# import noisereduce as nr
# import numpy as np

# BASE_DIR = r"D:\FYDP\FYDP DATASET"

# for pid in range(300, 403):
#     folder = os.path.join(BASE_DIR, f"{pid}_P")
#     input_path = os.path.join(folder, f"{pid}_patient.wav")
#     output_path = os.path.join(folder, f"{pid}_patient_denoised.wav")

#     if not os.path.exists(input_path):
#         print(f"Missing file: {input_path}")
#         continue

#     try:
#         # Load audio
#         audio, sr = sf.read(input_path)

#         # If stereo, take 1 channel
#         if audio.ndim > 1:
#             audio = audio[:, 0]

#         # Estimate noise from first 0.5 seconds (as fallback)
#         noise_sample = audio[:int(sr * 0.5)]

#         # Apply noise reduction
#         denoised_audio = nr.reduce_noise(y=audio, y_noise=noise_sample, sr=sr)

#         # Save denoised file
#         sf.write(output_path, denoised_audio, sr)
#         print(f"Denoised saved: {output_path}")

#     except Exception as e:
#         print(f"Error processing {pid}_P: {e}")












# This script extracts patient speech only segments from audio files based on transcripts.

# import os
# import pandas as pd
# import soundfile as sf
# import numpy as np

# BASE_DIR = r"D:\FYDP\FYDP DATASET"

# for pid in range(300, 403):
#     folder = os.path.join(BASE_DIR, f"{pid}_P")
#     wav_path = os.path.join(folder, f"{pid}_AUDIO.wav")
#     txt_path = os.path.join(folder, f"{pid}_TRANSCRIPT.csv")
#     out_path = os.path.join(folder, f"{pid}_patient_speechonly.wav")

#     if not os.path.exists(wav_path) or not os.path.exists(txt_path):
#         print(f"Missing files for: {pid}_P")
#         continue

#     try:
#         # Read the transcript, fix delimiter
#         df = pd.read_csv(txt_path, sep='\t')

#         # Fix column names if needed
#         if df.columns[0].startswith("start_time"):
#             df.columns = ["start", "stop", "speaker", "value"]

#         # Filter only speaking parts by participant
#         participant_segments = df[
#             (df['speaker'] == 'Participant') &
#             (df['value'].notnull()) &
#             (~df['value'].str.strip().isin(['', '[noise]', '[laughter]', '[silence]']))
#         ]

#         if participant_segments.empty:
#             print(f"No valid speaking segments in {pid}_P")
#             continue

#         # Read full audio
#         audio, sr = sf.read(wav_path)
#         if audio.ndim > 1:
#             audio = audio[:, 0]

#         # Extract and join all participant's speech segments
#         speech_audio = np.concatenate([
#             audio[int(float(row['start']) * sr): int(float(row['stop']) * sr)]
#             for _, row in participant_segments.iterrows()
#         ])

#         sf.write(out_path, speech_audio, sr)
#         print(f"Speech-only saved: {out_path}")

#     except Exception as e:
#         print(f"Error in {pid}_P: {e}")





# This script deletes the patient speech-only audio files if they exist.

# import os
# import glob

# BASE_DIR = r"D:\FYDP\FYDP DATASET"

# for pid in range(300, 403):
#     folder = os.path.join(BASE_DIR, f"{pid}_P")
#     speechonly_path = os.path.join(folder, f"{pid}_patient_speechonly.wav")

#     if os.path.exists(speechonly_path):
#         try:
#             os.remove(speechonly_path)
#             print(f"Deleted: {speechonly_path}")
#         except Exception as e:
#             print(f"Error deleting {speechonly_path}: {e}")
#     else:
#         print(f"Not found: {speechonly_path}")



# Need to install C++ build tools for webrtcvad to work








# import pandas as pd
# from pydub import AudioSegment
# import webrtcvad
# import numpy as np
# import noisereduce as nr
# import os
# import shutil

# # === CONFIG ===
# source_base_dir = r"D:\FYDP\FYDP DATASET"
# output_base_dir = r"E:\FYDP DATASET\daicwoz-dataset-patient-voice"

# vad_aggressiveness = 3  # 0-3 (most aggressive = 3)
# frame_duration_ms = 30
# buffer_ms = 10  # Padding around speech
# merge_threshold_ms = 5000  # 5 seconds

# def time_to_ms(t):
#     try:
#         return int(float(t) * 1000)
#     except:
#         return None

# def process_folder(folder_name):
#     print(f"\n[INFO] Processing folder: {folder_name}")

#     folder_path = os.path.join(source_base_dir, folder_name)
#     folder_prefix = folder_name[:-2] if folder_name.endswith('_P') else folder_name

#     audio_path = os.path.join(folder_path, f"{folder_prefix}_AUDIO.wav")
#     transcript_path = os.path.join(folder_path, f"{folder_prefix}_TRANSCRIPT.csv")

#     if not os.path.isfile(audio_path) or not os.path.isfile(transcript_path):
#         print(f"[WARN] Required files missing in {folder_path}")
#         return

#     out_folder = os.path.join(output_base_dir, folder_prefix)
#     os.makedirs(out_folder, exist_ok=True)
#     output_transcript_path = os.path.join(out_folder, f"{folder_prefix}_TRANSCRIPT.csv")

#     df_raw = pd.read_csv(transcript_path, sep="\t")
#     df_raw.columns = df_raw.columns.str.strip().str.lower()

#     speaker_counts = df_raw['speaker'].value_counts()
#     patient_speakers = [s for s in speaker_counts.index if str(s).strip().lower() != 'ellie']
#     if not patient_speakers:
#         print("[ERROR] No patient speaker found.")
#         return

#     patient_speaker = patient_speakers[0]
#     print(f"[INFO] Using '{patient_speaker}' as patient speaker.")

#     def is_clean_patient_voice(row):
#         speaker = str(row['speaker']).strip().lower()
#         value = str(row['value']).strip().lower()
#         if speaker != str(patient_speaker).strip().lower():
#             return False
#         if not value or any(tag in value for tag in ['[silence]', '[noise]', '[laughter]']):
#             return False
#         return True

#     df_raw['start_ms'] = df_raw['start_time'].apply(time_to_ms)
#     df_raw['end_ms'] = df_raw['stop_time'].apply(time_to_ms)
#     df = df_raw[df_raw.apply(is_clean_patient_voice, axis=1)].copy()

#     if df.empty:
#         print("[WARN] No clean patient segments found.")
#         return

#     original_audio = AudioSegment.from_wav(audio_path)
#     audio = original_audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
#     sample_rate = 16000
#     sample_width = 2
#     bytes_per_frame = int(sample_rate * frame_duration_ms / 1000) * sample_width
#     vad = webrtcvad.Vad(vad_aggressiveness)

#     def vad_split_segment(segment, global_start_ms):
#         raw = segment.raw_data
#         speech_ranges = []
#         in_speech = False
#         temp_start = None

#         for i in range(0, len(raw) - bytes_per_frame + 1, bytes_per_frame):
#             frame = raw[i:i + bytes_per_frame]
#             is_speech = vad.is_speech(frame, sample_rate)

#             if is_speech and not in_speech:
#                 in_speech = True
#                 temp_start = i
#             elif not is_speech and in_speech:
#                 in_speech = False
#                 speech_ranges.append((temp_start, i))

#         if in_speech:
#             speech_ranges.append((temp_start, len(raw)))

#         merged_ranges = []
#         for start, end in speech_ranges:
#             start_ms = global_start_ms + int((start / sample_width) * 1000 / sample_rate) - buffer_ms
#             end_ms = global_start_ms + int((end / sample_width) * 1000 / sample_rate) + buffer_ms
#             start_ms = max(0, start_ms)
#             end_ms = min(len(original_audio), end_ms)
#             merged_ranges.append((start_ms, end_ms))

#         return merged_ranges

#     # Step 1: Create full patient audio using VAD
#     full_patient_audio = AudioSegment.empty()
#     for _, row in df.iterrows():
#         start, end = row['start_ms'], row['end_ms']
#         if pd.isna(start) or pd.isna(end) or end <= start:
#             continue
#         segment = audio[start:end]
#         speech_times = vad_split_segment(segment, start)
#         for s_start, s_end in speech_times:
#             clip = original_audio[s_start:s_end]
#             samples = np.array(clip.get_array_of_samples()).astype(np.float32)
#             reduced = nr.reduce_noise(y=samples, sr=16000)
#             clean_clip = AudioSegment(
#                 reduced.astype(np.int16).tobytes(),
#                 frame_rate=16000,
#                 sample_width=2,
#                 channels=1
#             )
#             full_patient_audio += clean_clip

#     full_out_path = os.path.join(out_folder, f"PATIENT_AUDIO_{folder_prefix}.wav")
#     full_patient_audio.export(full_out_path, format="wav")
#     print(f"[✓] Exported full patient audio: {full_out_path}")

#     # Step 2: Chunk sentence-wise with merging < 5s
#     chunk_count = 0
#     chunk_log = []
#     pending_chunk = AudioSegment.empty()
#     pending_start = None
#     pending_text = []

#     for idx, row in df.iterrows():
#         start, end = row['start_ms'], row['end_ms']
#         value = str(row['value']).strip()
#         if pd.isna(start) or pd.isna(end) or end <= start:
#             continue

#         clip = original_audio[start:end]
#         samples = np.array(clip.get_array_of_samples()).astype(np.float32)
#         reduced = nr.reduce_noise(y=samples, sr=16000)
#         clean_clip = AudioSegment(
#             reduced.astype(np.int16).tobytes(),
#             frame_rate=16000,
#             sample_width=2,
#             channels=1
#         )

#         if pending_chunk.duration_seconds * 1000 < merge_threshold_ms:
#             if pending_start is None:
#                 pending_start = start
#             pending_chunk += clean_clip
#             pending_text.append(value)
#             continue
#         else:
#             chunk_count += 1
#             chunk_filename = f"{folder_prefix}_{chunk_count}_PATIENT_AUDIO.wav"
#             chunk_path = os.path.join(out_folder, chunk_filename)
#             pending_chunk.export(chunk_path, format="wav")
#             chunk_log.append({
#                 'Chunk Name': chunk_filename,
#                 'Start(ms)': pending_start,
#                 'End(ms)': start,
#                 'Duration(s)': round(pending_chunk.duration_seconds, 2),
#                 'Utterance': ' '.join(pending_text)
#             })
#             pending_chunk = clean_clip
#             pending_start = start
#             pending_text = [value]

#     if len(pending_chunk) > 0:
#         chunk_count += 1
#         chunk_filename = f"{folder_prefix}_{chunk_count}_PATIENT_AUDIO.wav"
#         chunk_path = os.path.join(out_folder, chunk_filename)
#         pending_chunk.export(chunk_path, format="wav")
#         chunk_log.append({
#             'Chunk Name': chunk_filename,
#             'Start(ms)': pending_start,
#             'End(ms)': end,
#             'Duration(s)': round(pending_chunk.duration_seconds, 2),
#             'Utterance': ' '.join(pending_text)
#         })

#     if chunk_log:
#         shutil.copy2(transcript_path, output_transcript_path)
#         df_log = pd.DataFrame(chunk_log)
#         log_csv_path = os.path.join(out_folder, f"{folder_prefix}_chunk_metadata.csv")
#         df_log.to_csv(log_csv_path, index=False)
#         print(f"[✓] Saved chunk metadata CSV: {log_csv_path}")
#     else:
#         print("[!] ERROR: No chunks were saved.")

# def main():
#     folders = [f for f in os.listdir(source_base_dir)
#                if os.path.isdir(os.path.join(source_base_dir, f)) and f.endswith('_P')]
#     print(f"[INFO] Found {len(folders)} folders to process.")
#     for folder in folders:
#         process_folder(folder)

# if __name__ == "__main__":
#     main()











# import pandas as pd
# from pydub import AudioSegment
# import webrtcvad
# import numpy as np
# import noisereduce as nr
# import os
# import shutil

# # === CONFIG ===
# source_base_dir = r"D:\FYDP\FYDP DATASET"
# output_base_dir = r"E:\FYDP DATASET\daicwoz-dataset-patient-voice"

# vad_aggressiveness = 3  # 0-3
# frame_duration_ms = 30
# buffer_ms = 10
# merge_threshold_ms = 5000  # 5 seconds

# def time_to_ms(t):
#     try:
#         return int(float(t) * 1000)
#     except:
#         return None

# def process_folder(folder_name):
#     print(f"\n[INFO] Processing folder: {folder_name}")

#     folder_path = os.path.join(source_base_dir, folder_name)
#     folder_prefix = folder_name[:-2] if folder_name.endswith('_P') else folder_name

#     audio_path = os.path.join(folder_path, f"{folder_prefix}_AUDIO.wav")
#     transcript_path = os.path.join(folder_path, f"{folder_prefix}_TRANSCRIPT.csv")

#     if not os.path.isfile(audio_path) or not os.path.isfile(transcript_path):
#         print(f"[WARN] Required files missing in {folder_path}")
#         return

#     out_folder = os.path.join(output_base_dir, folder_prefix)
#     os.makedirs(out_folder, exist_ok=True)
#     output_transcript_path = os.path.join(out_folder, f"{folder_prefix}_TRANSCRIPT.csv")

#     df_raw = pd.read_csv(transcript_path, sep="\t")
#     df_raw.columns = df_raw.columns.str.strip().str.lower()

#     speaker_counts = df_raw['speaker'].value_counts()
#     patient_speakers = [s for s in speaker_counts.index if str(s).strip().lower() != 'ellie']
#     if not patient_speakers:
#         print("[ERROR] No patient speaker found.")
#         return

#     patient_speaker = patient_speakers[0]
#     print(f"[INFO] Using '{patient_speaker}' as patient speaker.")

#     def is_clean_patient_voice(row):
#         speaker = str(row['speaker']).strip().lower()
#         value = str(row['value']).strip().lower()
#         if speaker != str(patient_speaker).strip().lower():
#             return False
#         if not value or any(tag in value for tag in ['[silence]', '[noise]', '[laughter]']):
#             return False
#         return True

#     df_raw['start_ms'] = df_raw['start_time'].apply(time_to_ms)
#     df_raw['end_ms'] = df_raw['stop_time'].apply(time_to_ms)
#     df = df_raw[df_raw.apply(is_clean_patient_voice, axis=1)].copy()

#     if df.empty:
#         print("[WARN] No clean patient segments found.")
#         return

#     original_audio = AudioSegment.from_wav(audio_path)
#     audio = original_audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
#     sample_rate = 16000
#     sample_width = 2
#     bytes_per_frame = int(sample_rate * frame_duration_ms / 1000) * sample_width
#     vad = webrtcvad.Vad(vad_aggressiveness)

#     def vad_split_segment(segment, global_start_ms):
#         raw = segment.raw_data
#         speech_ranges = []
#         in_speech = False
#         temp_start = None

#         for i in range(0, len(raw) - bytes_per_frame + 1, bytes_per_frame):
#             frame = raw[i:i + bytes_per_frame]
#             is_speech = vad.is_speech(frame, sample_rate)
#             if is_speech and not in_speech:
#                 in_speech = True
#                 temp_start = i
#             elif not is_speech and in_speech:
#                 in_speech = False
#                 speech_ranges.append((temp_start, i))

#         if in_speech:
#             speech_ranges.append((temp_start, len(raw)))

#         merged_ranges = []
#         for start, end in speech_ranges:
#             start_ms = global_start_ms + int((start / sample_width) * 1000 / sample_rate) - buffer_ms
#             end_ms = global_start_ms + int((end / sample_width) * 1000 / sample_rate) + buffer_ms
#             start_ms = max(0, start_ms)
#             end_ms = min(len(original_audio), end_ms)
#             merged_ranges.append((start_ms, end_ms))
#         return merged_ranges

#     # === FULL PATIENT AUDIO CREATION ===
#     full_patient_audio = AudioSegment.empty()
#     for _, row in df.iterrows():
#         start, end = row['start_ms'], row['end_ms']
#         if pd.isna(start) or pd.isna(end) or end <= start:
#             continue
#         segment = audio[start:end]
#         speech_times = vad_split_segment(segment, start)
#         for s_start, s_end in speech_times:
#             clip = original_audio[s_start:s_end]
#             samples = np.array(clip.get_array_of_samples()).astype(np.float32)
#             reduced = nr.reduce_noise(y=samples, sr=16000)
#             clean_clip = AudioSegment(
#                 reduced.astype(np.int16).tobytes(),
#                 frame_rate=16000,
#                 sample_width=2,
#                 channels=1
#             )
#             full_patient_audio += clean_clip

#     full_out_path = os.path.join(out_folder, f"PATIENT_AUDIO_{folder_prefix}.wav")
#     full_patient_audio.export(full_out_path, format="wav")
#     print(f"[✓] Exported full patient audio: {full_out_path}")

#     # === CHUNKING ===
#     chunk_count = 0
#     chunk_log = []
#     pending_chunk = AudioSegment.empty()
#     pending_start = None
#     pending_text = []
#     pending_indexes = []

#     def ms_to_minsec_ms(ms):
#         minutes = ms // 60000
#         seconds = (ms % 60000) // 1000
#         millis = ms % 1000
#         return f"{minutes:02d}:{seconds:02d}.{millis:03d}"

#     for idx, row in df.iterrows():
#         start, end = row['start_ms'], row['end_ms']
#         value = str(row['value']).strip()
#         if pd.isna(start) or pd.isna(end) or end <= start:
#             continue

#         clip = original_audio[start:end]
#         samples = np.array(clip.get_array_of_samples()).astype(np.float32)
#         reduced = nr.reduce_noise(y=samples, sr=16000)
#         clean_clip = AudioSegment(
#             reduced.astype(np.int16).tobytes(),
#             frame_rate=16000,
#             sample_width=2,
#             channels=1
#         )

#         # Merge short clips
#         if pending_chunk.duration_seconds * 1000 < merge_threshold_ms:
#             if pending_start is None:
#                 pending_start = start
#             pending_chunk += clean_clip
#             pending_text.append(value)
#             pending_indexes.append(idx)
#             continue
#         else:
#             chunk_count += 1
#             chunk_filename = f"{folder_prefix}_{chunk_count}_PATIENT_AUDIO.wav"
#             chunk_path = os.path.join(out_folder, chunk_filename)
#             pending_chunk.export(chunk_path, format="wav")

#             chunk_log.append({
#                 'Chunk Name': chunk_filename,
#                 'Start': ms_to_minsec_ms(pending_start),
#                 'End': ms_to_minsec_ms(start),
#                 'Duration (s)': round(pending_chunk.duration_seconds, 2),
#                 'Utterance': ' '.join(pending_text),
#                 'Transcript Indexes': '"' + ','.join(map(str, pending_indexes)) + '"',
#                 'Merged Count': len(pending_text)
# })

#             pending_chunk = clean_clip
#             pending_start = start
#             pending_text = [value]
#             pending_indexes = [idx]

#     # Final flush
#     if len(pending_chunk) > 0:
#         chunk_count += 1
#         chunk_filename = f"{folder_prefix}_{chunk_count}_PATIENT_AUDIO.wav"
#         chunk_path = os.path.join(out_folder, chunk_filename)
#         pending_chunk.export(chunk_path, format="wav")

#         chunk_log.append({
#             'Chunk Name': chunk_filename,
#             'Start': ms_to_minsec_ms(pending_start),
#             'End': ms_to_minsec_ms(end),
#             'Duration (s)': round(pending_chunk.duration_seconds, 2),
#             'Utterance': ' '.join(pending_text),
#             'Transcript Indexes': '"' + ','.join(map(str, pending_indexes)) + '"',
#             'Merged Count': len(pending_text)
# })

#     # === SAVE LOGS ===
#     if chunk_log:
#         shutil.copy2(transcript_path, output_transcript_path)
#         df_log = pd.DataFrame(chunk_log)
#         log_csv_path = os.path.join(out_folder, f"{folder_prefix}_chunk_metadata.csv")
#         df_log.to_csv(log_csv_path, index=False)
#         print(f"[✓] Saved chunk metadata CSV: {log_csv_path}")
#     else:
#         print("[!] ERROR: No chunks were saved.")

# def main():
#     folders = [f for f in os.listdir(source_base_dir)
#                if os.path.isdir(os.path.join(source_base_dir, f)) and f.endswith('_P')]
#     print(f"[INFO] Found {len(folders)} folders to process.")
#     for folder in folders:
#         process_folder(folder)

# if __name__ == "__main__":
#     main()


