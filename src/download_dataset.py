import os
import requests
import subprocess

# https://zenodo.org/records/5120004#.Yhxr0-jMJBA

# Define species and target base folder
species_list = ["Larus michahellis", "Columba livia", "Myiopsitta monachus", "Psittacula krameri", "Corvus cornix", "Falcus tinnunculus"]
base_folder = "sounds_32khz"

# Recordings from Xeno-canto
def get_xeno_canto_data(species):
    """Fetch metadata of recordings for a given species from Xeno-canto."""
    base_url = "https://www.xeno-canto.org/api/2/recordings"
    response = requests.get(f"{base_url}?query={species.replace(' ', '+')}")
    
    if response.status_code == 200:
        return response.json().get("recordings", [])
    else:
        print(f"Error fetching data for {species}")
        return []

# Download audio files into the respective species folder
def download_audio(recordings, species):
    """Download and convert recordings to WAV."""
    species_folder = os.path.join(base_folder, species.replace(" ", "_"))
    os.makedirs(species_folder, exist_ok=True)

    for record in recordings:
        file_url = f"https://www.xeno-canto.org/{record['id']}/download"
        mp3_filename = os.path.join(species_folder, f"{species.replace(' ', '_')}_{record['id']}.mp3")
        wav_filename = mp3_filename.replace(".mp3", ".wav")

        if not os.path.exists(wav_filename):
            print(f"Downloading: {mp3_filename}")
            try:
                audio_data = requests.get(file_url, stream=True)
                audio_data.raise_for_status()

                with open(mp3_filename, "wb") as file:
                    for chunk in audio_data.iter_content(1024):
                        file.write(chunk)

                # Convert MP3 to WAV
                # print(f"Converting to WAV: {wav_filename}")
                subprocess.run(["ffmpeg", "-i", mp3_filename, "-acodec", "pcm_s16le", "-ar", "32000", wav_filename], check=True)    # Change audio format and sampling rate as needed

                os.remove(mp3_filename)  # Delete MP3 after conversion
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {mp3_filename}: {e}")
        else:
            print(f"Already exists: {wav_filename}")

def main():
    for species in species_list:
        print(f"\nFetching recordings for {species}...")
        recordings = get_xeno_canto_data(species)
        if recordings:
            download_audio(recordings, species)
        else:
            print(f"No recordings found for {species}")

if __name__ == "__main__":
    main()
