import argparse
import librosa
import matplotlib.pyplot as plt
import numpy as np

def load_audio(audio_path):
    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    return audio_data, sample_rate

def save_spectrogram(audio_data, sample_rate):
    # Calculate the spectrogram
    S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)

    # Display the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    # Save the spectrogram as an image
    plt.savefig("spectrogram.png")
    print(f'Spectrogram saved as spectrogram.png')

def fourier_transform(audio_data, sample_rate):
    # Calculate the Fourier Transform
    stft = librosa.stft(audio_data)

    # Convert amplitude to dB (decibels)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    # Display the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(stft_db, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig("fourier_transform.png")
    print(f'Fourier transform saved as fourier_transform.png')

    return stft_db
    




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, required=True)
    args = parser.parse_args()

    audio_data, sample_rate = load_audio(args.audio)
    print('Audio data shape: {}'.format(audio_data.shape))
    print('Sample rate: {}'.format(sample_rate))

    save_spectrogram(audio_data, sample_rate)
    stft_db = fourier_transform(audio_data, sample_rate)

    


if __name__ == '__main__':
    main()
