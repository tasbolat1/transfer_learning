import numpy as np 
import torch
import matplotlib.pyplot as plt
import pickle

# num_epoch = 2000
# results_dict_file = "BioTac_info_400/results_dict.pkl"
# new_name = "BioTac_info_400/results_dict_smooth_" + str(num_epoch) + ".pkl"
# results_dict = pickle.load(open(results_dict_file,'rb'))
# pickle.dump(results_dict,open(new_name,'wb'))
# results = results_dict["results"]
# loss = results_dict["loss"]
# conf_mat = results_dict["conf_mat"]

# print(conf_mat)

# fig, ax = plt.subplots(figsize=(15, 7))
# epoch_num = np.arange(len(loss), dtype=np.int)
# ax.plot(epoch_num, loss, label="train:" + str(results[0])+ " test:" + str(results[1]))
# ax.set_xlabel('epoch')
# ax.set_ylabel('loss')
# ax.grid(True)
# plt.legend(loc='upper right')
# figname = "figures/smooth_" + str(num_epoch) + "_loss.png"
# plt.savefig(figname)
# plt.show()


# num_epoch = 10000
# results_dict_file = "BioTac_info_400/results_dict_Feb_loadLSTM_400_10000.pkl"
# # results_dict_file = "BioTac_info_400/results_dict_" + str(num_epoch) + ".pkl"
# # new_name = "BioTac_info_400/results_dict_smooth_" + str(num_epoch) + ".pkl"
# results_dict = pickle.load(open(results_dict_file,'rb'))
# # pickle.dump(results_dict,open(new_name,'wb'))
# print("keys", [key for key in results_dict])
# results = results_dict["results"]
# train_loss = results_dict["train_loss"]
# test_loss = results_dict["test_loss"]
# train_acc = results_dict["train_acc"]
# test_acc = results_dict["test_acc"]
# conf_mat = results_dict["conf_mat"]

# # print(conf_mat)
# assert len(train_loss) == len(test_loss), "different loss len, train: {}, test: {}".format(len(train_loss), len(test_loss))
# epoch_num = np.arange(len(train_loss), dtype=np.int)

# # if single plot
# fig, ax = plt.subplots(figsize=(15, 7))
# ax.plot(epoch_num, train_loss, label="train:" + str(results[0]))
# ax.plot(epoch_num, test_loss, label="test: " + str(results[1]))
# ax.set_xlabel('epoch')
# ax.set_ylabel('loss')
# ax.grid(True)
# plt.legend(loc='upper right')


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
# ax2.grid(True)

# ax2.set_xlabel('epoch')
# plt.legend(loc='upper right')





# compare two dicts
num_epoch = 1
# TODO:
dict_f1 = "BioTac_info/dict_BioTac_epoch_1.pkl"
# dict_f2 = "BioTac_info_400/results_dict_Feb1_10000.pkl"
dict1 = pickle.load(open(dict_f1,'rb'))
# dict2 = pickle.load(open(dict_f2,'rb'))

results1 = dict1["results"]
train_loss1 = dict1["train_loss"]
test_loss1 = dict1["valid_loss"]
train_acc1 = dict1["train_acc"]
test_acc1 = dict1["valid_acc"]
conf_mat1 = dict1["conf_mat"]

results2 = dict2["results"]
train_loss2 = dict2["train_loss"]
test_loss2 = dict2["valid_loss"]
train_acc2 = dict2["train_acc"]
test_acc2 = dict2["valid_acc"]
conf_mat2 = dict2["conf_mat"]
# print(results1)
# print(train_loss1, test_loss1, train_acc1, test_acc1)
# print(conf_mat1)

# assert len(train_loss) == len(test_loss), "different loss len, train: {}, test: {}".format(len(train_loss), len(test_loss))
epoch_num = np.arange(len(train_loss1), dtype=np.int)

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig, ax1 = plt.subplots(1, 1)
ax1.plot(epoch_num, train_loss1, 'b-', label="train w lstm")
ax1.plot(epoch_num, test_loss1, 'y-', label="test w lstm")
# ax1.plot(epoch_num, train_loss2, 'r--', label="train")
# ax1.plot(epoch_num, test_loss2, 'g--', label="test")
ax1.set_ylabel('loss')
plt.legend(loc='upper right')
ax1.grid(True)
ax1.set_xlabel('epoch')

# ax2.plot(epoch_num, train_acc1, 'b-', label="train w lstm "+ str(results1[0]))
# ax2.plot(epoch_num, test_acc1, 'y-', label="test w lstm "+ str(results1[1]))
# ax2.plot(epoch_num, train_acc2, 'r--', label="train "+ str(results2[0]))
# ax2.plot(epoch_num, test_acc2, 'g--', label="test "+ str(results2[1]))
# ax2.set_ylabel("acc")
# ax2.grid(True)

# ax2.set_xlabel('epoch')
# plt.legend(loc='lower right')


# # save the figure
# figname = "figures/with(out)_lstm_" + str(num_epoch) + "_loss_acc.png" 
# plt.savefig(figname)

plt.show()