import os
import random
import torchaudio
from torchaudio import transforms
from tqdm import tqdm
import pandas as pd


class FutureSupport(Exception):
    pass


def get_background_noise(file, duration, sameple_rate):
    # global BACKGROUNG_FILE, DURATION, SAMPLE_RATE
    chosen_file = random.choice(file)
    wave, sr = torchaudio.load(chosen_file)
    if sr != sameple_rate:
        resampler = transforms.Resample(orig_freq=sr, new_freq=sameple_rate)
        wave = resampler(wave)
    l = int(duration * sameple_rate)
    start_indx = random.randint(0, wave.shape[1] - l - 100)
    return wave[0:1, start_indx: start_indx + l], sr


def get_positive_label(positive_file):
    chosen_file = random.choice(positive_file)
    wave, sr = torchaudio.load(chosen_file)
    return wave[0:1, :], sr


#
# def get_negative_label(negative_file):
#     # global NEGATIVE_FILE
#     chosen_file = random.choice(negative_file)
#     wave, sr = torchaudio.load(chosen_file)
#     return wave, sr
#

def create_new_sample(background, label, stretche_speed=False):
    if stretche_speed:
        raise FutureSupport("")
    label_len = label.shape[1]
    background_len = background.shape[1]
    start_index_insert = random.randint(0, background_len - label_len - 100)

    background_reduce = random.uniform(0.0, 1)
    background *= background_reduce
    ratio = random.uniform(0.1, 0.6)
    background[:, start_index_insert: start_index_insert + label_len] = background[:,
                                                                        start_index_insert: start_index_insert + label_len] * ratio + (
                                                                                1 - ratio) * label[:, :]
    return background


if __name__ == "__main__":
    SAVE_DIR = "../data/train_sample_v3"
    try:
        os.mkdir(SAVE_DIR)
    except:
        pass

    NUMBER_TRAINING_SAMPLE = 40000
    NUMBER_TESTING_SAMPLE = 10000
    NUMBER_CHUCKS = (NUMBER_TESTING_SAMPLE + NUMBER_TRAINING_SAMPLE) // 1000

    for chuck in range(NUMBER_CHUCKS):
        try:
            os.mkdir(f"{SAVE_DIR}/chuck_{chuck}")
        except:
            pass

    DURATION = 1.5
    SAMPLE_RATE = 16000
    with open("../data/train.txt", "r") as f:
        ls_file = f.readlines()
        POSITIVE_FILE_TRAIN = [os.path.join("../data/speech_commands_v0.02", i.strip()) for i in ls_file]

    with open("../data/test.txt", "r") as f:
        ls_file = f.readlines()
        POSITIVE_FILE_TEST = [os.path.join("../data/speech_commands_v0.02", i.strip()) for i in ls_file]

    BACKGROUND_ROOT_PATH = "../data/speech_commands_v0.02/_background_noise_rebase"
    BACKGROUNG_FILE = [os.path.join(BACKGROUND_ROOT_PATH, file) for file in os.listdir(BACKGROUND_ROOT_PATH) if
                       file.endswith(".wav")]
    # background_csv = []
    # for i in range(10000):
    #     chunk_id = i % NUMBER_CHUCKS
    #     background, _ = get_background_noise(BACKGROUNG_FILE, DURATION, SAMPLE_RATE)
    #     file_name = f"chuck_{chunk_id}/Audio_background_{i}.wav"
    #     background_csv.append((file_name, 0))
    #     torchaudio.save(os.path.join(SAVE_DIR, file_name))
    #
    # df = pd.DataFrame(background_csv)
    # df.to_csv(os.path.join("../data", "background_noise.csv"), index=False)

    train_csv_file = []
    test_csv_file = []
    for i in tqdm(range(NUMBER_TRAINING_SAMPLE)):
        chunk_id = i % NUMBER_CHUCKS
        background, _ = get_background_noise(BACKGROUNG_FILE, DURATION, SAMPLE_RATE)
        is_positive = random.choice([0, 1, 2])
        label = 1 if is_positive == 0 else 0
        if label:
            voice, _ = get_positive_label(POSITIVE_FILE_TRAIN)
            out = create_new_sample(background, voice)
        else:
            out = background
        file_name = f"chuck_{chunk_id}/Audio{label}_{i}_train.wav"
        train_csv_file.append((file_name, label))
        torchaudio.save(os.path.join(SAVE_DIR, file_name), out, SAMPLE_RATE)

    for i in tqdm(range(NUMBER_TESTING_SAMPLE)):
        chunk_id = i % NUMBER_CHUCKS
        background, _ = get_background_noise(BACKGROUNG_FILE, DURATION, SAMPLE_RATE)
        is_positive = random.choice([0, 1, 2])
        label = 1 if is_positive == 0 else 0
        if label:
            voice, _ = get_positive_label(POSITIVE_FILE_TEST)
            out = create_new_sample(background, voice)
        else:
            out = background
        file_name = f"chuck_{chunk_id}/Audio{label}_{i}_test.wav"
        test_csv_file.append((file_name, label))
        torchaudio.save(os.path.join(SAVE_DIR, file_name), out, SAMPLE_RATE)

    df = pd.DataFrame(train_csv_file)
    df.to_csv(os.path.join("../data", "train_csv_v3.csv"), index=False)

    df = pd.DataFrame(test_csv_file)
    df.to_csv(os.path.join("../data", "test_csv_v3.csv"), index=False)
