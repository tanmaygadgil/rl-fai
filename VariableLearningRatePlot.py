import pandas as pd
import os
import json
import matplotlib.pyplot as plt


json_df = None
fig, axs = plt.subplots(2, 3)
fig.suptitle("Effect of Learning Rate on performance")
axs[0, 0].set_title("Learning Rate = 0.01")
axs[0, 1].set_title("Learning Rate = 0.001")
axs[0, 2].set_title("Learning Rate = 0.0001")
axs[1, 0].set_title("Learning Rate = 0.01")
axs[1, 1].set_title("Learning Rate = 0.001")
axs[1, 2].set_title("Learning Rate = 0.0001")
results_dic = {}
color_dic = {"dqn": red, "ddqn": blue, ""}

for env in ['cartpole', 'lunarlander-discrete']:
    for n in [1, 2]:
        for g in [0.95, 0.9, 0.8]:
            for lr in [0.01, 0.001, 0.0001]:
                results_dic[(env, n, g, lr)] = 0


for ax in axs.flat:
    ax.set(xlabel='episodes', ylabel='rewards')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

for subdir, dirs, files in os.walk('results 2'):
    for file in files:
        csv_df = None
        fileName = os.path.join(subdir, file)
        if fileName.endswith('json'):
            data = json.load(open(fileName))
            json_df = pd.DataFrame(data, index = [0])
            continue
        elif fileName.endswith("csv"):
            csv_df = pd.read_csv(fileName).rolling(15).mean()
        else:
            continue

        if json_df.loc[0, "episodes"] == 600:
            continue

        lab = ""
        env = json_df.loc[0, "environment"]
        lr = json_df.loc[0, "lr"]
        n = json_df.loc[0, "network"]
        g = json_df.loc[0, "gamma"]

        if env == "cartpole":
            if lr == 0.01 and results_dic[(env, n, g, lr)] != 1:
                axs[0, 0].plot(csv_df.loc[1:, 'episodes'], csv_df.loc[1:, 'rewards'], label =f"network = {n:d}, gamma = {g:.2f}", color = color_dic[(n,g)])
                results_dic[(env, n, g, lr)] = 1
            elif lr == 0.001 and results_dic[(env, n, g, lr)] != 1:
                axs[0, 1].plot(csv_df.loc[1:, 'episodes'], csv_df.loc[1:, 'rewards'], label = f"network = {n:d}, gamma = {g:.2f}", color = color_dic[(n,g)])
                results_dic[(env, n, g, lr)] = 1
            elif lr == 0.0001 and results_dic[(env, n, g, lr)] != 1:
                axs[0, 2].plot(csv_df.loc[1:, 'episodes'], csv_df.loc[1:, 'rewards'], label = f"network = {n:d}, gamma = {g:.2f}", color = color_dic[(n,g)])
                results_dic[(env, n, g, lr)] = 1

        elif env == "lunarlander-discrete":
            if lr == 0.01 and results_dic[(env, n, g, lr)] != 1:
                axs[1, 0].plot(csv_df.loc[1:, 'episodes'], csv_df.loc[1:, 'rewards'], label = f"network = {n:d}, gamma = {g:.2f}", color = color_dic[(n,g)])
                results_dic[(env, n, g, lr)] = 1
            elif lr == 0.001 and results_dic[(env, n, g, lr)] != 1:
                axs[1, 1].plot(csv_df.loc[1:, 'episodes'], csv_df.loc[1:, 'rewards'], label = f"network = {n:d}, gamma = {g:.2f}", color = color_dic[(n,g)])
                results_dic[(env, n, g, lr)] = 1
            elif lr == 0.0001 and results_dic[(env, n, g, lr)] != 1:
                axs[1, 2].plot(csv_df.loc[1:, 'episodes'], csv_df.loc[1:, 'rewards'], label = f"network = {n:d}, gamma = {g:.2f}", color = color_dic[(n,g)])
                results_dic[(env, n, g, lr)] = 1


axs[0, 0].legend(loc='lower right', prop={'size': 5})
axs[0, 1].legend(loc='lower right', prop={'size': 5})
axs[0, 2].legend(loc='lower right', prop={'size': 5})
axs[1, 0].legend(loc='upper right', prop={'size': 5})
axs[1, 1].legend(loc='lower right', prop={'size': 5})
axs[1, 2].legend(loc='lower right', prop={'size': 5})
plt.show()
