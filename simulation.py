import numpy as np
import torch
import matplotlib.pyplot as plt
from models.model import BcResNetModel
from utils.generals import trigger_word_detect, convet2logmel
import pyaudio
from queue import Queue
import time


def callback(in_data, frame_count, time_info, status):
    global run, timeout, data, fs
    if time.time() > timeout:
        run = False
    data0 = np.frombuffer(in_data, dtype='int16')
    data = np.append(data, data0)
    if len(data) > feed_samples:
        data = data[-feed_samples:]
        q.put(data)
    return in_data, pyaudio.paContinue


model = BcResNetModel(n_class=2, scale=1.5, dropout=0.1)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.DataParallel(model).to(device)
ckpt = torch.load("runs/KWSexp3/checkpoint_best.pth.taz", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

chunk_duration = 0.1
fs = 8000
chunk_samples = int(fs * chunk_duration)
feed_duration = 1.5
feed_samples = int(fs * feed_duration)

assert feed_duration / chunk_duration == int(feed_duration / chunk_duration)

q = Queue()

plot_prob = Queue()
for i in range(50):
    plot_prob.put(0.0)

run = True
timeout = time.time() + 3 * 60
data = np.zeros(feed_samples, dtype='float32')

fig, ax = plt.subplots(1, figsize=(15, 7))

stream = pyaudio.PyAudio().open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=fs,
    input=True,
    frames_per_buffer=chunk_samples,
    input_device_index=0,
    stream_callback=callback
)
line, = ax.plot(range(plot_prob.qsize()), list(plot_prob.queue), '-', lw=2)
stream.start_stream()
last = time.time()
try:
    while True:
        data = torch.from_numpy(q.get().reshape(1, -1))
        spectrum = convet2logmel(data, fs).unsqueeze(0)
        preds, prob = trigger_word_detect(model, spectrum, device)
        plot_prob.get()
        plot_prob.put(prob)
        line.set_ydata(list(plot_prob.queue))
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()
        if preds:
            if time.time() - last == 2:
                last = time.time()
                # pass

except (KeyboardInterrupt, SystemExit):
    stream.stop_stream()
    stream.close()
    run = False
