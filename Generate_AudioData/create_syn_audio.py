""" 
This module is used to create synthetic data overlaying enviromental noises
with drones noises.
"""


#Need a mix of both with drones and without drones, those should be labeled
#Need a wide range of drone noise types
#Will be focusing on nature sounds overlayed with drone sounds
#Need drone sounds of varying intensity
#Need to add mix multiple tracks and also add periods of silence / very low waves
#Need to make the tracks several minutes long to establish a moving baseline
#Need to label which track is the normal vsa drone sound

""" 
*Each segment will be a "x" minute snippet of the entire recording, 
the dtype will prob be a list of the WAV rawdata vals.
 - 15 sec segments , 15 min audios? 

*Need to decide how long those snippets will be.

*The drone segment needs to be an overlay of the 
drone sound plus the segments drone_indx +-1.

*Create multiple datasets with varying drone sound intensities to mimick proximity
*Figure out how the sound mixing will occur with the drone
"""




#Params : Length of the new audio
import logging
import librosa
import numpy as np
import json
from pathlib import Path
import json
import itertools
import soundfile as sf


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DJ_splice:
    def __init__(self):
        self.NON_URBAN_TOC = Path("Data/Audio_Files/Clips/non_urban_toc.json")
        self.URBAN_TOC     = Path("Data/Audio_Files/Clips/urban_toc.json")
        self.raw_audio_subfolder = "Shortened_Base_Audios"
        self.Update_Table_of_Content() #Update audio file table of content referencing raw audio file name
        

    #Updates audio file table of content on init.
    def Update_Table_of_Content(self,) -> None:
        non_urban_files = list(Path(f"Data/Audio_Files/Raw_Audios/{self.raw_audio_subfolder}/NonUrban").glob("*"))
        urban_files     = list(Path(f"Data/Audio_Files/Raw_Audios/{self.raw_audio_subfolder}/Urban").glob("*"))

        self.non_urban_toc = self._update_toc(non_urban_files, self.NON_URBAN_TOC)
        self.urban_toc     = self._update_toc(urban_files, self.URBAN_TOC)
        logger.info("Audio file table of content has been updated")

    #Main method
    def PrepareData(self, toc_file_path: Path, Urban_nonUbran_bool: bool, anomaly_list: list, overlayed_2_original_ratio: float, segment_duration: int):        
        """
        toc_file_path -> file path to table of content
        Urban_nonUrban_bool -> Urban vs NonUrban indicator
        anomaly_list -> anomalies to parse [Explosion, Gunshots, etc]
        overlayed_2_original_ratio -> ratio of anomaly clips : total clips
        segment_duration -> in seconds, length of a single mini clip of base audio
        """
        
        # Check if files are for urban or nonUrban audios
        if Urban_nonUbran_bool == 0:
            logger.info("Files will be saved to the NonUrban Directory")
            while True:
                user_input = input("Type y to continue n to cancell")
                if user_input.lower() == "n":
                    return 1 #error
                elif user_input.lower() == "y":
                    break
                else:
                    continue
        elif Urban_nonUbran_bool == 1:
            logger.info("Files will be saved to the Urban Directory")
            while True:
                user_input = input("Type y to continue n to cancell")
                if user_input.lower() == "n":
                    return 1 #error
                elif user_input.lower() == "y":
                    break
                else:
                    continue
        else:
            logger.warning("Invalid input for Urban_nonUrban bool param, terminating process")
            return 1 #error

        #load json
        try:
            with open(toc_file_path, 'r') as f:
                files_dict = json.load(f)
        except FileNotFoundError:
            logger.error(f"{toc_file_path}, TOC file does not exist")
            return 1
        total_files = len(files_dict) 

        error_files = [] #Unprocessed files with issues
        if Urban_nonUbran_bool == 0:
            category = "NonUrban"
        else:
            category = "Urban"

        #for every base audio file
        progress_cntr = 0 #files processed
        for file_name in files_dict:
            audio_path = Path(f"./Data/Audio_Files/Raw_Audios/{self.raw_audio_subfolder}/{category}/{file_name}")
            #load audio
            try:
                base_y , base_sr = librosa.load(audio_path) #base audio data
            except Exception as e:
                logger.warning(f"Error: {e}\n Audio file will be skipped")
                error_files.append(audio_path)
                continue
            
            #make directory for base audio and its variation
            audio_dir_str = f"./Data/Audio_Files/Clips/{category}/{files_dict[file_name]}"
            audio_dir_path = Path(audio_dir_str).mkdir(parents=True, exist_ok=True)    
            for anomaly_type in anomaly_list: #Explosion, Gunshots, Animals, etc: Note these folders must exist prior
                #Quick check if anomaly_type and path file is correct
                anomaly_path = Path(f"./Data/Audio_Files/Raw_Audios/Anomalies/{anomaly_type}")
                if not anomaly_path.exists():
                    logger.error(f"Anomaly path does not exist: {anomaly_path}")
                    return 1 #error
               
                #Make subfolder for each anomaly sound for each base audio #Creating data for every anomaly type base audio combo
                anomaly_dir_str = audio_dir_str+f"/{anomaly_type}"
                anomaly_dir = Path(anomaly_dir_str).mkdir(parents=True, exist_ok=True)


                #overlay base and anomaly audio randomly
                data = self.Create_Audio_Data(base_y, anomaly_type, base_sr, overlayed_2_original_ratio, segment_duration)
                new_amplitudes = list(itertools.chain.from_iterable(data[0])) 
                
                #save info in json
                final_json = {}
                final_json["base_audio_file_name"] = file_name 
                final_json["segment_durations"] = segment_duration 
                final_json["anomaly_to_original_ratio"] = overlayed_2_original_ratio
                final_json["anomaly_info"] = data[1] #Will contain tuples (anomaly segment index, scaling factor of anomaly "loudness")
                final_json["anomaly_type"] = anomaly_type
                final_json["sample_rate"] = base_sr #same for anomaly and base audio
                
                json_file_name = f"{files_dict[file_name]}_{anomaly_type}_clips.json"

                #Check if file exists, if you want to overwrite
                json_path = Path(anomaly_dir_str+f"/{json_file_name}")
                if json_path.exists():
                    while True:
                        user_input = input(f"{json_path} already exists, type 'y' to overwrite type 'n' to skip file")
                        if user_input.lower() == 'y' or user_input.lower() == 'n':
                            break
                        else:
                            continue
                    if user_input.lower() == 'n': 
                        continue
                    else:
                        logger.warning(f"Overwriting existing file: {json_path}")
                        with open(json_path, 'w') as f:
                            json.dump(final_json, f, indent=4)

                        #Saving new audio
                        audio_file_name = f"{files_dict[file_name]}_{anomaly_type}_clips.wav"
                        audio_file_path = Path(anomaly_dir_str+f"/{audio_file_name}")
                        sf.write(audio_file_path, new_amplitudes, base_sr) 
                else:
                    #Saving new audio
                    audio_file_name = f"{files_dict[file_name]}_{anomaly_type}_clips.wav"
                    audio_file_path = Path(anomaly_dir_str+f"/{audio_file_name}")
                    sf.write(audio_file_path, new_amplitudes, base_sr)
                    #Save meta data
                    with open(json_path, 'w') as f:
                        json.dump(final_json, f, indent=4) 
            progress_cntr+=1
            logger.info(f"Base audio processed: {progress_cntr}/{total_files}")

        
        logger.info(f"Process is complete\n{len(error_files)} files were skipped:")
        for bad_files in error_files:
            logger.info(f"{bad_files}") 
        
    
    #Clip and Overlay a single base audio, returns list of ordered segment and indices of segment that were overlayed
    def Create_Audio_Data(self, original_audio: np.array, anomaly_type: str, sample_rate: int, overlayed_2_original_ratio: float, segment_duration: int = 10) -> list[list[np.array], list[int]]:
        """
        orginal_audio ->  array of amplitude vals
        anomaly_type -> subfolder name of anomalies e.g (Animals, Explosion, etc) , note its case sensitive
        sample_rate -> assumes all use librosa default rate
        overlayed_2_original_ratio -> ratio of anomaly clips : total clips
        segment_duration -> in seconds, length of a single mini clip of base audio
        """
        anomaly_path = Path(f"./Data/Audio_Files/Raw_Audios/Anomalies/{anomaly_type}")

        if not anomaly_path.exists():
            logger.error(f"Anomaly path does not exist: {anomaly_path}")
            return []

        anomaly_files = list(anomaly_path.glob("*"))

        if not anomaly_files:
            logger.warning(f"No anomaly files found in: {anomaly_path}")
            return []

        #Break up audio into uniform clips
        base_clip_list = self._Segment_Audio(original_audio, segment_duration, sample_rate)
        
        num_of_overlayed_clips = int(len(base_clip_list) * overlayed_2_original_ratio)
        
        # Use choice with replace=False to prevent the same clip being overlayed multiple times
        overlayed_index_lst = np.random.choice(len(base_clip_list), size=num_of_overlayed_clips, replace=False)
        
        #Choose random loudness of anomaly
        anomaly_weights = np.random.uniform(0.01, .5, len(overlayed_index_lst))
        anomaly_info = list(zip(overlayed_index_lst.tolist(), anomaly_weights.tolist())) #Will contain tuples (anomaly segment index, scaling factor of anomaly "loudness")
        

        for info in anomaly_info:
            #Choose random anomaly variation
            anomaly_clip, sr = librosa.load(anomaly_files[np.random.randint(0, len(anomaly_files))])
            
            #Overlay the base clip with the anomaly
            mix_clip = self._Overlay_Audio(base_clip_list[info[0]], anomaly_clip, sr, info[1])
            
            #Update the base_clip_list
            base_clip_list[info[0]] = mix_clip
        
        return [base_clip_list, anomaly_info] #base clips with anomalies overlayed and their indices
        


#-------------------- Helper Functions -----------------------
    #Overlay two audio so as if they occured simultaneously
    def _Overlay_Audio(self, base_clip: np.array, anomaly_clip: np.array, sample_rate: int, anomaly_weight: float) -> np.array:
        """
        base_clip -> base audio / background
        anomaly_clip -> clip that will add noise to the data
        sample_rate -> sample rate is chosen to be 22050Hz for all audios
        anomaly_weight -> scaling factor for anomaly audio to control loudness balance ranges [0.01,1].   
        """
        # Trim or pad anomaly to match base length
        target_len = len(base_clip)
        if len(anomaly_clip) >= target_len:
            new_clip = anomaly_clip[:target_len]
        else:
            new_clip = np.pad(anomaly_clip, (0, target_len - len(anomaly_clip)))

        mixed = (base_clip) + (anomaly_weight * new_clip)
        # Normalize to prevent clipping (exceedng amplitude max range  [-1,1])
        if np.max(np.abs(mixed)) > 1.0:
            mixed = mixed / np.max(np.abs(mixed))
        
        return mixed #Overlayed audio
    
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
    
    #update table of content helper func.
    def _update_toc(self, files, toc_path) -> dict:
        # Ensure directory exists
        toc_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing TOC if it exists
        if toc_path.exists():
            with open(toc_path, "r") as f:
                toc = json.load(f)
        else:
            toc = {}

        # Append only new files
        for file in files:
            if file.name not in toc:
                toc[file.name] = f"audio_{len(toc) + 1}"

        # Save updated TOC
        with open(toc_path, "w") as f:
            json.dump(toc, f, indent=4)

        return toc



if __name__ == "__main__":
    pass