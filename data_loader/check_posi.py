# import os
# from pathlib import Path
# from tqdm import tqdm
# import pandas as pd
#
# root = "../data/train_sample_v2"
# train_df = pd.read_csv("../data/train_csv_final.csv")
# test_df = pd.read_csv("../data/test_csv_final.csv")
#
# for i in tqdm(range(len(train_df))):
#     path = train_df.iloc[i, 0]
#     if Path(os.path.join(root, path)).is_file():
#         continue
#     print(f"Cannot find: {path}")
#
# for i in tqdm(range(len(test_df))):
#     path = test_df.iloc[i, 0]
#     if Path(os.path.join(root, path)).is_file():
#         continue
#     print(f"Cannot find: {path}")
#


import os
import pandas as pd

test_file = "../data/speech_commands_v0.02/testing_list.txt"
vali_file = "../data/speech_commands_v0.02/validation_list.txt"

eval = []
with open(test_file, "r") as f:
    test_file = f.readlines()
    eval.extend([i.strip() for i in test_file if "happy" in i])

with open(vali_file, "r") as f:
    test_file = f.readlines()
    eval.extend([i.strip() for i in test_file if "happy" in i])

total = list(os.listdir("../data/speech_commands_v0.02/happy"))
total = [f"happy/{i}" for i in total]

train = list(set(total) - set(eval))

df_train = pd.DataFrame(train)
df_eval = pd.DataFrame(eval)
df_train.to_csv("../data/train.txt", index=False)
df_eval.to_csv("../data/test.txt", index=False)
