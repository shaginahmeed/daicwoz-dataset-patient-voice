# DAIC-WOZ Patient Voice Extractor

This repo provides a Python script (`chunk_audio_code.py`) along with dataset (300-402) to process audio and transcript data from the DAIC-WOZ dataset. Its primary goal is to extract clean, patient-only speech segments, apply noise reduction, and then segment these into meaningful chunks suitable for downstream analysis or machine learning model training.


* **Batch Processing:** Processes multiple participant folders from a specified source directory.
* **Transcript-Guided Filtering:** Filters utterances based on speaker (excluding 'Ellie') and removes segments explicitly tagged as `[silence]`, `[noise]`, or `[laughter]`.
* **Voice Activity Detection (VAD):** Utilizes `webrtcvad` for precise identification of speech within the filtered transcript segments, enhancing accuracy by removing implicit silences.
* **Noise Reduction (NR):** Applies noise reduction using the `noisereduce` library to suppress background noise present within the detected speech segments.
* **Full Patient Audio Export:** Generates a single, continuous WAV file containing all processed patient speech for each participant.
* **Intelligent Chunking:** Divides the full patient audio into smaller, contextually relevant chunks. Short utterances (below a configurable threshold, default 5 seconds) are merged with subsequent ones to create more complete "expressions."


* **Detailed Chunk Metadata:** Creates a CSV file for each participant, logging details about the exported chunks, including:
    * `Chunk Name`: Filename of the exported audio chunk.
    * `Start(s)`: Start time of the chunk in seconds relative to the original audio.
    * `End(s)`: End time of the chunk in seconds relative to the original audio.
    * `Duration(s)`: Total duration of the chunk in seconds.
    * `Utterance`: Concatenated text from the transcript segments that form the chunk.
    * `Transcript Indexes`: Original row indices from the raw transcript that were merged into this chunk.
    * `Merged Count`: Number of individual transcript segments merged to form this chunk.


  **Packages:**
    ```bash
    pip install pandas pydub webrtcvad numpy noisereduce soundfile
    python -m pip install --upgrade pip setuptools --force-reinstall
    ```

## DAIC-WOZ Dataset Insights (from Provided Metadata)

Based on the `train_split_Depression_AVEC2017.csv` and `dev_split_Depression_AVEC2017.csv` files in online, here's an overview of the depression status for participants with IDs from **300 to 402**:

* **Total participants found in this range with depression labels (from Training and Development Splits):** 77
    * **Depressed (PHQ8_Binary = 1):** 30 participants
    * **Not Depressed (PHQ8_Binary = 0):** 47 participants

* **Participant IDs in this range with labels:**
    `[302, 303, 304, 305, 307, 310, 312, 313, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 330, 331, 333, 335, 336, 338, 339, 340, 341, 343, 344, 345, 346, 347, 348, 350, 351, 352, 353, 355, 356, 357, 358, 360, 362, 363, 364, 366, 367, 368, 369, 370, 371, 372, 374, 375, 376, 377, 379, 380, 381, 382, 383, 385, 386, 388, 389, 390, 391, 392, 393, 395, 397, 400, 401, 402]`

* **Participants in the test split within this range (no depression labels available in these files):**
    `[300, 301, 306, 308, 309, 311, 314, 323, 329, 332, 334, 337, 349, 354, 359, 361, 365, 373, 378, 384, 387, 396, 399]`