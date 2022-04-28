import pandas as pd
import os
import json
import matplotlib.pyplot as plt


json_df = None
fig, axs = plt.subplots(2, 2)
fig.suptitle("Effect of Learning Rate on performance")
axs[0, 0].set_title("Network = 1")
axs[0, 1].set_title("Network = 2")
axs[1, 0].set_title("Network = 1")
axs[1, 1].set_title("Network = 2")
# lab1 = "lr = 0.01, gamma = 0.95"
# lab2 = "lr = 0.001, gamma = 0.99"
# lab3 = "lr = 0.0001, gamma = 0.8"
# lab4 = "network = 2, gamma = 0.95"
# lab5 = "network = 2, gamma = 0.99"
# lab6 = "network = 2, gamma = 0.8"
results_dic = {}

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

        # if lr == .01:
        #     if g == 0.95:
        #         lab = lab1
        #     elif g == 0.99:
        #         lab = lab2
        #     else:
        #         lab = lab3
        # else:
        #     if g == 0.95:
        #         lab = lab4
        #     elif g == 0.99:
        #         lab = lab5
        #     else:
        #         lab = lab6

        if env == "cartpole":
            if n == 1 and results_dic[(env, n, g, lr)] != 1:
                axs[0, 0].plot(csv_df.loc[1:, 'episodes'], csv_df.loc[1:, 'rewards'], label = "lr = %d, gamma = %d"%(lr, g))
                results_dic[(env, n, g, lr)] = 1
            elif n == 2 and results_dic[(env, n, g, lr)] != 1:
                axs[0, 1].plot(csv_df.loc[1:, 'episodes'], csv_df.loc[1:, 'rewards'], label = "lr = %d, gamma = %d"%(lr, g))
                results_dic[(env, n, g, lr)] = 1

        elif env == "lunarlander-discrete":
            if n == 1 and results_dic[(env, n, g, lr)] != 1:
                axs[1, 0].plot(csv_df.loc[1:, 'episodes'], csv_df.loc[1:, 'rewards'], label = "lr = %d, gamma = %d"%(lr, g))
                results_dic[(env, n, g, lr)] = 1
            elif lr == 0.001 and results_dic[(env, n, g, lr)] != 1:
                axs[1, 1].plot(csv_df.loc[1:, 'episodes'], csv_df.loc[1:, 'rewards'], label = lab)
                results_dic[(env, n, g, lr)] = 1
            elif lr == 0.0001 and results_dic[(env, n, g, lr)] != 1:
                axs[1, 2].plot(csv_df.loc[1:, 'episodes'], csv_df.loc[1:, 'rewards'], label = lab)
                results_dic[(env, n, g, lr)] = 1


axs[0, 0].legend(loc='lower right', prop={'size': 5})
axs[0, 1].legend(loc='lower right', prop={'size': 5})
axs[0, 2].legend(loc='lower right', prop={'size': 5})
axs[1, 0].legend(loc='upper right', prop={'size': 5})
axs[1, 1].legend(loc='lower right', prop={'size': 5})
axs[1, 2].legend(loc='lower right', prop={'size': 5})
plt.show()
