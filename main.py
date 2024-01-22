import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

def load_audio(audio_path):
    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    return audio_data, sample_rate

def save_spectrogram(audio_data, sample_rate, filename):
    # Calculate the spectrogram
    S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)

    # Display the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    # Save the spectrogram as an image
    plt.savefig(filename)
    print(f'Spectrogram saved as {filename}')

def fourier_transform(audio_data, sample_rate):
    # Calculate the Fourier Transform
    stft = librosa.stft(audio_data)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, required=True)
    args = parser.parse_args()

    audio_data, sample_rate = load_audio(args.audio)
    print('Audio data shape: {}'.format(audio_data.shape))
    print('Sample rate: {}'.format(sample_rate))

    save_spectrogram(audio_data, sample_rate, "spectrogram.png")


if __name__ == '__main__':
    main()
