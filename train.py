from soulflow import *
import pandas as pd
import cv2
import time
import matplotlib.pylab as plt


def cross_entropy_error(y:np.ndarray, t:np.ndarray):
    return -np.sum(t * np.log(y + 1e-7))


def label_to_tensor(N_id:int, label:str, tensor:np.ndarray):
    if (label == "happiness"):
        tensor[N_id, 0] = 1
    if (label == "surprise"):
        tensor[N_id, 1] = 1
    if (label == "others"):
        tensor[N_id, 2] = 1
    if (label == "disgust"):
        tensor[N_id, 3] = 1
    if (label == "fear"):
        tensor[N_id, 4] = 1
    if (label == "repression"):
        tensor[N_id, 5] = 1
    if (label == "sadness"):
        tensor[N_id, 6] = 1


def image_warp(img:np.ndarray):
    H = 480
    W = 640
    rand = np.random.randint(40, size=6)
    src = np.float32([[0, 0], [W-1, 0], [0, H-1]])
    dst = np.float32([[rand[0], rand[1]], [W-1-rand[2], rand[3]], [rand[4], H-1-rand[5]]])
    M1 = cv2.getAffineTransform(src, dst)
    M2 = np.float32([[1, 0, np.random.rand()*200-100], [0, 1, np.random.rand()*40-20]])
    imgt = cv2.warpAffine(img, M1, (W, H))
    imgt = cv2.warpAffine(imgt, M2, (W, H))
    imgt = cv2.resize(imgt, [256, 192])
    return imgt


def train_gru(n):
    excel = pd.read_excel("CASME2-coding-20190701.xlsx")
    emotion = excel["Estimated Emotion"]
    subject = excel["Subject"]
    file_name = excel["Filename"]
    OnsetFrame = excel["OnsetFrame"]
    OffsetFrame = excel["OffsetFrame"]
    print("Sample count:", len(subject))
    sf = SoulFlow(1, "soulflow.pkl")
    sf.setGradientToZero()
    print("Init finished.")
    loss_list = []
    label = np.zeros([1, 7], dtype=np.float32)
    t1 = time.time()
    for t in range(n):
        print("Step:", t)
        sf.resetTemp()
        sample_id = np.random.randint(100, len(subject))
        label *= 0
        label_to_tensor(0, emotion[sample_id], label)
        if subject[sample_id] < 10:
            sub_str = "0" + str(subject[sample_id])
        else:
            sub_str = str(subject[sample_id])
        dir = "CASME2-RAW/sub" + sub_str + "/" + file_name[sample_id]
        onset = OnsetFrame[sample_id]
        offset = OffsetFrame[sample_id]
        img_n = 0
        tt1 = time.time()
        for n in range(onset - 1, offset + 1):
            path = dir + "/img" + str(n) + ".jpg"
            img = image_warp(cv2.imread(path))
            sf.set_sample(0, img)
            sf.forward_gru(1)
            img_n += 1
        sf.deviceSync()
        tt2 = time.time()
        print("----------------------------------------------------------------------------------------")
        print("----", img_n / (tt2 - tt1), "images per second.")
        result = sf.predict_gru()
        print(result)
        loss = cross_entropy_error(result, label)
        print("loss:", loss)
        if np.isnan(loss):
            sf.free()
            exit(-1)
        loss_list.append(loss)
        gradient = result - label
        sf.backward_gru(gradient)
        if t % 16 == 0:
            sf.momentum_gru(0.5, 0.0005)
            sf.setGradientToZero()
        print("----------------------------------------------------------------------------------------")
    t2 = time.time()
    print("Duration:", t2 - t1)
    x = np.arange(len(loss_list))
    plt.plot(x, loss_list)
    plt.show()
    sf.show_histogram()
    sf.deviceMemUsage()
    sf.to_storage("soulflow.pkl")
    sf.free()


def test_gru(n):
    excel = pd.read_excel("CASME2-coding-20190701.xlsx")
    emotion = excel["Estimated Emotion"]
    subject = excel["Subject"]
    file_name = excel["Filename"]
    OnsetFrame = excel["OnsetFrame"]
    OffsetFrame = excel["OffsetFrame"]
    print("Sample count:", len(subject))
    sf = SoulFlow(1, "soulflow.pkl")
    sf.setGradientToZero()
    print("Init finished.")
    label = np.zeros([1, 7], dtype=np.float32)
    c = 0
    for t in range(n):
        print("Step:", t)
        sf.resetTemp()
        sample_id = np.random.randint(100, len(subject))
        # sample_id = np.random.randint(1, 100)
        label *= 0
        label_to_tensor(0, emotion[sample_id], label)
        if subject[sample_id] < 10:
            sub_str = "0" + str(subject[sample_id])
        else:
            sub_str = str(subject[sample_id])
        dir = "CASME2-RAW/sub" + sub_str + "/" + file_name[sample_id]
        onset = OnsetFrame[sample_id]
        offset = OffsetFrame[sample_id]
        img_n = 0
        tt1 = time.time()
        for i in range(onset - 1, offset + 1):
            path = dir + "/img" + str(i) + ".jpg"
            img = image_warp(cv2.imread(path))
            sf.set_sample(0, img)
            sf.forward_gru(1)
            img_n += 1
        sf.deviceSync()
        tt2 = time.time()
        print("----------------------------------------------------------------------------------------")
        print("----", img_n / (tt2 - tt1), "images per second.")
        result = sf.predict_gru().reshape([7])
        label = label.reshape([7])
        print(result)
        print(label)
        if (result.argmax() == label.argmax()):
            c += 1
        label = label.reshape([1, 7])
        print("----------------------------------------------------------------------------------------")
    sf.free()
    print("Correct:", c, "accuracy:", c / n)


def random_image(excel:pd.DataFrame, N_id:int, labels:np.ndarray):
    emotion = excel["Estimated Emotion"]
    subject = excel["Subject"]
    file_name = excel["Filename"]
    apexFrame = excel["ApexFrame"]
    sample_id = np.random.randint(100, len(subject))
    label = emotion[sample_id]
    label_to_tensor(N_id, label, labels)
    if subject[sample_id] < 10:
        sub_str = "0" + str(subject[sample_id])
    else:
        sub_str = str(subject[sample_id])
    dir = "CASME2-RAW/sub" + sub_str + "/" + file_name[sample_id]
    apex = apexFrame[sample_id]
    path = dir + "/img" + str(apex) + ".jpg"
    img = cv2.imread(path)
    img = image_warp(img)
    return img


def train_image(n:int):
    batch_size = 32
    excel = pd.read_excel("CASME2-coding-20190701.xlsx")
    # None "soulflow.pkl"
    sf = SoulFlow(batch_size, None)
    print("Init finished.")
    loss_list = []
    labels = np.zeros([batch_size, 7], dtype=np.float32)
    for t in range(n):
        print("Step:", t)
        sf.setGradientToZero()
        labels *= 0
        for i in range(batch_size):
            img = random_image(excel, i, labels)
            sf.set_sample(i, img)
        # print(labels)
        result = sf.forward_image()
        loss = cross_entropy_error(result, labels)
        print("    loss:", loss)
        if np.isnan(loss):
            sf.free()
            exit(-1)
        loss_list.append(loss)
        gradient = result - labels
        sf.backward_image(gradient)
        sf.momentum_image(0.8, 0.0005)
    x = np.arange(len(loss_list))
    plt.plot(x, loss_list)
    plt.show()
    sf.show_histogram()
    sf.deviceMemUsage()
    sf.to_storage("soulflow.pkl")
    sf.free()


def test_image(n:int):
    excel = pd.read_excel("CASME2-coding-20190701.xlsx")
    emotion = excel["Estimated Emotion"]
    subject = excel["Subject"]
    file_name = excel["Filename"]
    apexFrame = excel["ApexFrame"]
    sf = SoulFlow(1, "soulflow.pkl")
    label = np.zeros([1, 7], dtype=np.float32)
    c = 0
    for t in range(n):
        sample_id = np.random.randint(100, len(subject))
        # sample_id = np.random.randint(1, 100)
        if subject[sample_id] < 10:
            sub_str = "0" + str(subject[sample_id])
        else:
            sub_str = str(subject[sample_id])
        dir = "CASME2-RAW/sub" + sub_str + "/" + file_name[sample_id]
        print(t, "id:", sample_id, "Sample:", dir)
        apex = apexFrame[sample_id]
        path = dir + "/img" + str(apex) + ".jpg"
        img = cv2.imread(path)
        img = image_warp(img)
        sf.set_sample(0, img)
        result = sf.forward_image().reshape([7])
        label *= 0
        label_to_tensor(0, emotion[sample_id], label)
        label = label.reshape([7])
        print(result)
        print(label)
        if (result.argmax() == label.argmax()):
            c += 1
        label = label.reshape([1, 7])
    sf.free()
    print("Correct:", c, "accuracy:", c / n)


def print_model():
    with open("soulflow.pkl", 'rb') as f:
        W_list = pickle.load(f)
        print(W_list)


def show_hist(self):
    sf = SoulFlow(1, "soulflow.pkl")
    sf.show_histogram()
    sf.free()


if __name__ == '__main__':
    # train_image(4000)
    # test_image(1000)
    # train_gru(20000)
    test_gru(100)
