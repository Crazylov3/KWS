import torchaudio


wave, sr = torchaudio.load("happy.wav")
print(wave.shape)
