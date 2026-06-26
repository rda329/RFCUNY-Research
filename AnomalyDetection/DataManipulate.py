"""
This module implements a class that 
handles all data transformations / manipulations
"""
import logging
import librosa
import numpy as np
from pathlib import Path
import os
import glob
from sklearn.model_selection import train_test_split
import json
import pandas as pd

#Print statements system
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DataChef():
    def __init__(self,):
        pass
    
    #Pipeline to prepare data
    #Cuts the original audio into uniform segments specified by meta data json
    #Transforms each audio segment to frequency domain using discrete fourier transformation
    #Normalizes all audios
    #stores all data in pandas dataframe, each segmenet is a row
    def PrepData_Method(self, file_path: str, window_size: int) -> pd.DataFrame:
        """ 
        file_path: audio file path
        segment_duration: length of audio segements in secs
        window_size: number of values in a window to average when denoising, -> see _preprocess_audio method
        """
        logger.info(f"Starting data preparation | file: '{file_path}' | window_size: {window_size}")
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"Audio file not found: '{file_path}'")
            raise FileNotFoundError(f"Audio file not found: '{file_path}'")

        #load audio json
        json_path = file_path.with_suffix(".json")
        if not json_path.exists():
            logger.error(f"JSON metadata file not found: '{json_path}'")
            raise FileNotFoundError(f"JSON metadata file not found: '{json_path}'")

        with open(json_path, "r") as f:
            data = json.load(f)
        segment_duration = data["segment_durations"]
        anomaly_info = data["anomaly_info"]
        anomaly_seg_indices = []
        for val in anomaly_info:
            anomaly_seg_indices.append(val[0])
        #logger.info(f"Loaded metadata | segment_duration: {segment_duration}s | anomalies flagged: {len(anomaly_seg_indices)} {anomaly_seg_indices}")

        #load audio data
        #logger.info(f"Loading audio file: '{file_path}'")
        y, sr = librosa.load(Path(file_path))  # Bug fix: librosa is a module, not callable — use librosa.load()
        #logger.info(f"Audio loaded | sample_rate: {sr} | total_samples: {len(y)} | duration: {len(y)/sr:.2f}s")

        #Create uniform audio segments
        segment_lst = self._Segment_Audio(y, segment_duration, sr)
        #logger.info(f"Audio segmented | total_segments: {len(segment_lst)}")

        if len(segment_lst) == 0:
            logger.warning("No segments were created — check segment_duration or audio length")

        #df to store processed audio segments
        audio_df = {
            "segment_index": [],
            "anomaly_bool": [],    # 1 for anomaly else 0
            "processed_audio": [],  # Resulting audio data after processing
        }
        
        anomaly_count = 0
        for index, segment in enumerate(segment_lst):
            #logger.debug(f"Processing segment {index + 1}/{len(segment_lst)}")
            processed_audio = self._preprocess_audio(segment, window_size)

            if processed_audio is None or len(processed_audio) == 0:
                logger.warning(f"Segment {index} returned empty/None after preprocessing — skipping")
                continue

            audio_df["segment_index"].append(index)
            audio_df["processed_audio"].append(processed_audio)
            if index in anomaly_seg_indices:
                audio_df["anomaly_bool"].append(1)
                anomaly_count += 1
            else:
                audio_df["anomaly_bool"].append(0)

        #logger.info(f"Preprocessing complete | segments_processed: {len(audio_df['segment_index'])} | anomaly_segments: {anomaly_count}")

        unexpected = [i for i in anomaly_seg_indices if i >= len(segment_lst)]
        if unexpected:
            logger.warning(f"Anomaly indices out of segment range (possible segment_duration mismatch): {unexpected}")

        audio_df = pd.DataFrame(audio_df)
        #Each value of the audio data is a feature
        expanded = pd.DataFrame(audio_df['processed_audio'].tolist(), index=audio_df.index)
        expanded.columns = [f'{i}' for i in range(len(expanded.columns))]
        audio_df = audio_df.join(expanded)
        audio_df = audio_df.drop(columns=['processed_audio'])


        """ 
        Notes about dataframe
        - Each row is a audio segment from the same audio file
        - cols 0 - N are amplitudes transformed using a discrete fourier transformation
        - segmemt_index is the index corresponding to the original order 
        - anomaly_bool is a boolean which indicates 1 if the segment contains anomaly or 0 if not.
        """

        return audio_df


    #Get specified audio file paths
    def GetAudios(self, natural_bool: bool, anomaly_type: str,) :
        """
        natural_bool: NonUrban 0 , Urban 1
        anomaly_type: Gunshot, Explosion, UAV, etc
        """
        # Normalize anomaly_type to be case-insensitive
        anomaly_type_normalized = anomaly_type.strip().lower()

        # Map natural_bool to environment folder name 
        if natural_bool:
            environment = "Urban"
        else: 
            environment = "NonUrban"

        # Base path: .\Data\Audio_Files\Clips\<Urban|NonUrban>\
        base_dir = os.path.join(".", "Data", "Audio_Files", "Clips", environment)

        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Environment directory not found: '{base_dir}'")

        # Search pattern: .\Data\Audio_Files\Clips\<env>\audio_*\<anomaly_type>\*
        audio_extensions = '.wav'
        filtered_files = {}

        # Iterate over audio_1, audio_2, ... audio_n subdirectories
        for audio_subdir in os.listdir(base_dir):
            audio_subdir_path = os.path.join(base_dir, audio_subdir)
            if not os.path.isdir(audio_subdir_path):
                continue

            # Iterate over anomaly type subdirectories (case-insensitive match)
            for anomaly_subdir in os.listdir(audio_subdir_path):
                if anomaly_subdir.strip().lower() != anomaly_type_normalized:
                    continue

                anomaly_path = os.path.join(audio_subdir_path, anomaly_subdir)
                if not os.path.isdir(anomaly_path):
                    continue

                # Collect all matching audio files
                for file in os.listdir(anomaly_path):
                    if os.path.splitext(file)[1].lower() in audio_extensions:
                        filtered_files[audio_subdir] = os.path.join(anomaly_path, file)

        if not filtered_files:
            raise ValueError(
                f"No audio files found for environment='{environment}' "
                f"and anomaly_type='{anomaly_type}'"
            )

        return filtered_files


#------------------------------- Helper Functions ------------------------------------------------------
    #Preprocess audio file
    #Transform audio file into normalized frequency spectrum (Discrete Fourier Transformation using Fast Fourier Algo numpy implementation)
    #https://medium.com/@davidegrimaldi92/when-audio-speaks-differently-a-deep-dive-into-audio-anomaly-detection-with-machine-learning-7b81f2033959
    def _preprocess_audio(self, audio_segment: np.ndarray, window_size: int):
        """
        audio_segment: raw amplitude values of array segment
        window_size: number of values in a window to average when denoising
        """
        
        #MAKE 10 SEC SEGMENT WINDOWS, SAVE AS INDIVIUAL LISTS, SAVE IN PANDAS 
        #AFTER TRANSFORMING NORMALIZE THEN ALL TOGETHER BUT KEEP TRACK OF DIFFERENT SEGMENTS

        """ 
        Update this code we need to compare the the different segments within an audio file not 
        audio files between eachother
        """


        # Denoise using moving average
        denoised_audio = np.convolve(audio_segment, np.ones(window_size)/window_size, mode='valid')
        
        # Transform to frequency domain
        fft_result = np.fft.fft(denoised_audio)
        magnitude = np.abs(fft_result)[:len(fft_result) // 2]
        
        # Normalize by maximum amplitude, we care about the shape of the frequency not the "loudness"
        magnitude = magnitude / np.max(magnitude)
        
        return magnitude[1:] #excluding DC component (The average amplitude of the entire audio) provided in index 0
         

    #Method to get snippets of original audio
    def _ClipAudio(self, original_audio: np.array, sample_rate: int, start_sec: int = 0, duration_sec: int = 1, truncate_bool : bool = 1) -> np.array:
        """
        original_audio -> amplitude values from audio file
        sample_rate -> number of amplitudes sampled per sec
        start_sec, duration_sec -> start and length of audio clip
        truncate_bool -> 0 : return remaining audio if 0 elif 1 do not truncate if upperbound exceeds audio, return none
        """
        total_samples = len(original_audio)
        lower_bnd = start_sec * sample_rate
        upper_bnd = lower_bnd + duration_sec * sample_rate if duration_sec > 0 else total_samples

        if lower_bnd < 0 or lower_bnd >= total_samples:
            raise ValueError(f"start_sec={start_sec} is out of bounds for audio of length {total_samples / sample_rate:.2f}s")

        if upper_bnd > total_samples:
            logger.warning(f"Requested clip exceeds audio length")
            if truncate_bool == 0:
                logger.info("Returning remaining length")
                return original_audio[lower_bnd:] #return remaining len
            else: 
                logger.info("Returning empty array")
                return None

        return original_audio[lower_bnd:upper_bnd] #returns clip of audio
    
    #Segment an audio into uniform length clips
    def _Segment_Audio(self, original_audio: np.array, segment_duration: int, sample_rate: int):
        """
        original_audio -> amplitude values from audio file
        segment_duration -> duration in seconds of each clip
        sample_rate -> number of amplitudes sampled per sec
        """
        clip_lst = [] #Each clip in order 
        samples_per_segment = segment_duration * sample_rate
        num_of_segments = int(len(original_audio) / samples_per_segment)
        start_sec = 0 #start time in seconds of the clip

        for i in range(num_of_segments):
            clip = self._ClipAudio(original_audio, sample_rate, start_sec, segment_duration)
            if clip is not None:
                clip_lst.append(clip)
            start_sec += segment_duration
        return clip_lst #Return list of uniform length clips in order
    
    
    
    # ADD CROSS VALIDATION , Evaluate with cross-validation later for more robust evaluat


if __name__ == "__main__":
    pass