import numpy as np 
import torch
import matplotlib.pyplot as plt
import pickle

num_epoch = 2000
basename = "BioTac_info_400/results_dict_Feb_"
tws = [10, 20, 50, 100, 200, 300, 400]
train_accs = []
test_accs = []
for tw in tws:
    results_dict_file = basename + str(tw) + "_" + str(num_epoch) + ".pkl"
    print("process ", results_dict_file)
    results_dict = pickle.load(open(results_dict_file,'rb'))

    print("keys", [key for key in results_dict])
    results = results_dict["results"]
    train_accs.append(results[0])
    test_accs.append(results[1])
    # train_loss = results_dict["train_loss"]
    # test_loss = results_dict["test_loss"]
    # train_acc = results_dict["train_acc"]
    # test_acc = results_dict["test_acc"]
    # conf_mat = results_dict["conf_mat"]

# if single plot
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(tws, train_accs, label="train acc")
ax.plot(tws, test_accs, label="test acc")
ax.set_xlabel('time window')
ax.set_ylabel('acc')
ax.grid(True)
plt.legend(loc='upper right')

# # if double plots
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
# # # make a little extra space between the subplots
# # fig.subplots_adjust(hspace=0.5)
# ax1.plot(epoch_num, train_loss, label="train loss")
# ax1.plot(epoch_num, test_loss, label="test loss")
# ax1.set_ylabel('loss')
# ax1.grid(True)
# ax2.plot(epoch_num, train_acc, label="train acc "+ str(results[0]))
# ax2.plot(epoch_num, test_acc, label="test acc "+ str(results[1]))
# ax2.set_ylabel("acc")
# ax1.grid(True)

# ax2.set_xlabel('epoch')
# plt.legend(loc='upper right')


# save the figure
figname = "figures/comapare_diff_tw_epoch_" + str(num_epoch) + ".png" 
plt.savefig(figname)
plt.show()