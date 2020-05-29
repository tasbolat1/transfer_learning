'''
Three orthogonal metrics:
average motor current, replaced with average joint torque of a6 
instead of subtracting background noise from power, use an IIR filter to smoothen the signal before calculating power
for fineness, how to calculate f in eqn.18? I just use freqs in fftfreq multiplied by frate (2200)
Naive Bayes classifier ref: https://dzone.com/articles/naive-bayes-tutorial-naive-bayes-classifier-in-pyt

running tip: activate `opencv` conda env to use updated sklearn for plot_confusion_matrix
'''
import numpy as np 
import matplotlib.pyplot as plt
import csv
import copy
from scipy import signal
import time
import math
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import seaborn as sns


def build_dataset(filename):
    with open(filename, mode='w') as datafile:
        data_writer = csv.writer(datafile, delimiter=',')
        # data_writer.writerow(['MatName', 'SampleNum', 'Traction', 'Roughness', 'Fineness'])
        data_writer.writerow(['Traction', 'Roughness', 'Fineness', 'MatName'])
        

    for matNum in range(1, 22):
        for sampleNum in range(1, 51):
            # matNum = 10
            # sampleNum = 10
            matName = "mat" + str(matNum)
            print("process for ", matName, str(sampleNum))

            # obtain avg joint-torque-a6 for traction
            t = []
            ta6 = []
            with open("/home/ruihan/Documents/material_data_Feb/" + matName + "/" + matName + "_" + str(sampleNum) + "_jt.csv", 'r') as jt_infile:
                csv_reader = csv.reader(jt_infile, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        # print("Column names are " + ", ".join(row))
                        pass
                    else:
                        if line_count == 1:
                            t0 = float(row[0])

                        t.append(float(row[0])-t0)
                        ta6.append(float(row[6]))

                    line_count += 1
            traction = np.mean(np.array(ta6))
            # print('Processed %d lines for ta6.' % line_count, " ta6 len ", len(ta6), " avg ", traction)

            # obtain pac for roughness and fineness
            t = []
            pac = []
            with open("/home/ruihan/Documents/material_data_Feb/" + matName + "/" + matName + "_" + str(sampleNum) + "_bio.csv", 'r') as bio_infile:
                csv_reader = csv.reader(bio_infile, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        # print("Column names are" + ", ".join(row))
                        pass
                    else:
                        if line_count == 1:
                            t0 = float(row[0])

                        t.append(float(row[0])-t0)
                        # convert string to float
                        pac.extend([float(i) for i in row[4:26]])
                        
                    
                    line_count += 1
            pac = np.array(pac)
            # print('Processed %d lines for pac.' % line_count, pac.shape)
            # print("t len ", len(t), len(pac))  #800 17600
        
            # obtain power for roughness
            # apply 20-700Hz digital band-pass filter (66th order FIR filter)
            numtaps = 66+1
            cutoff = [20, 700]
            bp_filter = signal.firwin(numtaps, cutoff, pass_zero=False, fs=2200)
            # apply filter to signal
            pac_signal = pac[np.newaxis, :]
            # print(pac.shape, bp_filter.shape, bp_filter[np.newaxis, :].shape) #(1, 17600) (67,) (1, 67)
            tstart = time.time()
            pac_fil = signal.convolve(pac_signal, bp_filter[np.newaxis, :], mode='valid')
            conv_time = time.time() - tstart
            # print("conv use time ", conv_time)
            pac_fil = np.squeeze(pac_fil)
            # print("signal len ", len(pac),len(pac_fil)) # 17600 17534

            # noise reduction (IIR filter)
            n = 15  # the larger n is, the smoother curve will be
            b = [1.0 / n] * n
            a = 1
            pac_red = signal.lfilter(b,a,pac_fil)

            pac_power = np.sum(np.power(pac_red, 2))/len(pac_red)
            roughness = np.log(pac_power)
            # print("power", pac_power, "roughness", roughness)
            
            # # plot the org and filtered signal
            # fig, (ax_orig, ax_win, ax_filt, ax_red) = plt.subplots(4, 1, sharex=False)
            # t = np.arange(len(pac))
            # ax_orig.plot(t, pac)
            # ax_orig.set_ylabel('pac')
            # ax_orig.set_xlabel('Timestamp')
            # ax_orig.set_title('Original pac signal')
            # ax_orig.margins(0, 0.1)

            # ax_win.plot(bp_filter)
            # ax_win.set_title('Filter impulse response')
            # ax_win.margins(0, 0.1)

            # t_fil = np.arange(len(pac_fil))
            # ax_filt.plot(t_fil, pac_fil)
            # ax_filt.set_title('Filtered signal')
            # ax_filt.margins(0, 0.1)

            # t_red = np.arange((len(pac_red)))
            # ax_red.plot(t_red, pac_red)
            # ax_red.set_title("Noise reduction")
            # ax_red.margins(0, 0.1)

            # fig.tight_layout()
            # plt.savefig("fitlered_pac_" + matName + "_" + str(sampleNum) + ".png")
            # fig.show()


            # obtain spectral centroid (SC) for fineness
            pac_fft = np.fft.fft(pac)
            freqs = np.fft.fftfreq(len(pac))
            # print(freqs.min(), freqs.max()) 
            frate = 2200

            # convert the freq to Hz
            freqs_in_hertz = np.fabs(np.multiply(freqs, frate))
            # print("freqs in hertz", freqs_in_hertz)
            pac_fft2 = np.square(np.absolute(pac_fft))
            SC = np.sum(np.multiply(pac_fft2, freqs_in_hertz))/np.sum(pac_fft2)
            fineness = np.log(SC)
            # print("SC", SC, "fineness", fineness)

            with open(filename, mode='a+') as datafile:
                data_writer = csv.writer(datafile, delimiter=',')
                # data_writer.writerow([matName, sampleNum, traction, roughness, fineness])
                data_writer.writerow([traction, roughness, fineness, matNum-1])



def loadCsv(filename):
    lines = csv.reader(open(filename, mode='r'))
    dataset = list(lines)[1:]
    print("sample data", dataset[0])
    # convert data to float
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
        dataset[i][-1] = int(dataset[i][-1])
    return dataset

def splitDataset(dataset, testRatio, num_class):
    random.seed(1)
    testSize = int(len(dataset)*testRatio)
    testSize_class = int(testSize/num_class)
    print("test size per class", testSize_class)
    testSet = []
    test_idx = random.sample(range(50), testSize_class) # non-repeating
    print("test_idx", test_idx, len(dataset))
    test_idxes = []
    for i in range(num_class):
        test_idxes.extend([x+i*50 for x in test_idx])
    # reverse the order so that it does not mess up the idx
    test_idxes.sort(reverse=True)
    print(test_idxes)

    trainSet = dataset.copy()
    for test_id in test_idxes:
        testSet.append(trainSet.pop(test_id))
    print("train len", len(trainSet), "test len", len(testSet)) # 840, 210

    return trainSet, testSet


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg, 2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def loadDF(filename):
    df = pd.read_csv(filename, header=0, index_col=False)
    label = df['MatName']
    data = df.drop(labels=['MatName'], axis=1)
    print("data", data.shape, "label", label.shape) # data (1050, 3) label (1050,)
    return data, label


def main():
    datafile_name = "Bayes_data.csv"
    # build_dataset(datafile_name)
    # dataset =loadCsv(datafile_name)
    num_class = 21

    ''' Naive Bayes classifier '''
    # # [trainSet, testSet] = splitDataset(dataset, 0.2, num_class) # for implementation of NB from scratch, TODO
    # [data, target] = loadDF(datafile_name)
    # # split train&test dataset
    # X_train, X_test, y_train, y_test = train_test_split(data, target,  test_size=0.2, random_state=0)
    # model = GaussianNB() # acc 0.74
    # # model = MultinomialNB() # 0.41

    # # # without splitting ds
    # # model.fit(data, target)
    # # expected = target
    # # predicted = model.predict(data)
    # # # get acc and statistics
    # # print(metrics.classification_report(expected, predicted))
    # # print(metrics.confusion_matrix(expected, predicted))

    # # with splitting ds
    # y_pred = model.fit(X_train, y_train).predict(X_test)
    # print(metrics.classification_report(y_test, y_pred))
    # # print(metrics.confusion_matrix(y_test, y_pred)) # 0.72

    # # plot the conf matrix
    # title_options = [("Confusion matrix, without normalization", None),
    #                  ("Normalized confusion matrix", "true")]
    # for title, normalize in title_options:
    #     disp = metrics.plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues, normalize=normalize)
    #     disp.ax_.set_title(title)

    #     # print(title)
    #     # print(disp.confusion_matrix)
        
    # plt.show()

    ''' Plot the features for visualization '''
    # Traction	Roughness	Fineness	MatName
    df = pd.read_csv(datafile_name, index_col=False, header=0)
    print(df.head())
    sns.set(style='ticks', color_codes=True)

    # # create three plots separately
    # traction_plot = sns.catplot(x='MatName', y='Traction', data=df)
    # traction_plot.savefig("traction_plot.png")
    # roughness_plot = sns.catplot(x='MatName', y='Roughness', data=df)
    # roughness_plot.savefig("roughness_plot.png")
    # fineness_plot = sns.catplot(x='MatName', y='Fineness', data=df)
    # fineness_plot.savefig("fineness_plot.png")

    # combine three plots into one
    fig, axs = plt.subplots(nrows=3, sharex=True)
    print(axs)
    sns.catplot(x='MatName', y='Traction', data=df, ax=axs[0])
    sns.catplot(x='MatName', y='Roughness', data=df, ax=axs[1])
    sns.catplot(x='MatName', y='Fineness', data=df, ax=axs[2])
    plt.savefig("hand_craft_features_plot.png") # TODO: check how to remove extra figures
    plt.show()


if __name__ == "__main__":
    main()


