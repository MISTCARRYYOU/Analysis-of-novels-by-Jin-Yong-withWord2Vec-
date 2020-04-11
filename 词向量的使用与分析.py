from gensim.models import Word2Vec
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN



from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 设置matplotlib可以显示汉语
mpl.rcParams['axes.unicode_minus'] = False


def reduce_dimensions(model, vocabulary):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    vectors = [] # positions in vector space
    labels = [] # keep track of words to label our data again later
    for word in vocabulary:
        try:
            vectors.append(model.wv[word])
            labels.append(word)
        except KeyError:
            pass
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    # reduce using t-SNE
    vectors = np.asarray(vectors)
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


def plot_with_matplotlib(x_vals, y_vals, labels, selected_indices):
    visited = []
    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals, s=2)
    indices = list(range(len(x_vals)))[selected_indices[0]:selected_indices[1]]
    for i in indices:
        if labels[i] in visited:
            continue
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
        plt.scatter(x_vals[i], y_vals[i], s=2, color="red")
        visited.append(labels[i])
    plt.title("金庸小说高频率词向量关系降维结果图")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig("图片/金庸小说人物词向量关系降维结果图.png", dpi=600)
    plt.show()


def readtxt(path):
    with open(path, "r", encoding="utf-8") as f:
        cor = [eve.strip("\n") for eve in f]
    return cor


# labels为聚类结果的标签值
# forclusterlist为聚类所使用的样本集
# 函数的功能是将forclusterlist中的样本集按照labels中的标签值重新排序，得到按照类簇排列好的输出结果
def labels_to_original(labels, forclusterlist):
    assert len(labels) == len(forclusterlist)
    maxlabel = max(labels)
    numberlabel = [i for i in range(0, maxlabel + 1, 1)]
    numberlabel.append(-1)
    result = [[] for i in range(len(numberlabel))]
    for i in range(len(labels)):
        index = numberlabel.index(labels[i])
        result[index].append(forclusterlist[i])
    return result



if __name__ == "__main__":
    model = Word2Vec.load('./W2Vmodel/model')
    # print(model.most_similar(positive=['赵敏'], topn=5))
    # print(model.wv.vocab)
    # print(model['赵敏'])
    # print(model['徐铮'])
    # print(model.similarity('马春花','徐铮'))

    # 1-绘制人名词向量空间展示
    ORDER = 10 # 每个小说统计其频率前十的人名，共14*ORDER个人名
    vocabulary_ = []
    vocabulary = []
    for eve in readtxt("文本文档/全小说各自的人名词频统计.txt"):
        temp = eve.strip("\n").split()
        vocabulary_.append(temp[:ORDER])
    vocabulary_ = np.array(vocabulary_).T.tolist()
    for i in range(len(vocabulary_)):
        vocabulary += vocabulary_[i]

    vocabulary = [eve if len(eve) > 1 else "123" for eve in readtxt("文本文档/词频前一千词.txt")]
    vocabulary.remove("123")
    x_vals, y_vals, labels = reduce_dimensions(model, vocabulary)
    plot_with_matplotlib(x_vals, y_vals, labels, [0, 40])

    # 2-计算各个小说的主人公的距离最近的词向量聚类
    main_people = ['郭靖', '黄蓉', '小龙女', '杨过', '张无忌', '赵敏', '周芷若', '乔峰',
                   '段誉', '阿朱', '王语嫣', '韦小宝', '令狐冲', '盈盈', '苗若兰', '苗人凤',
                   '田归农', '胡一刀', '陈家洛', '狄云', '水笙', '石破天', '叮当', '胡斐',
                   '苗人凤', '程灵素', '袁承志', '袁冠南', '林玉龙', '任飞燕',
                   '李文秀', '苏普']
    data = []
    for name in main_people:
        temp = model.most_similar(positive=[name], topn=5)
        data.append([name] + [str(eve[0]) + "(" + str(round(eve[1],3)) + ")" for eve in temp])

    df = pd.DataFrame(data)
    df.to_excel("文本文档/主人公相似度.xlsx")

    # 3-高频词语的聚类分析
    word = readtxt("文本文档/词频前一千词.txt")
    weight = []
    for eve in word:
        weight.append(model[eve])
    # 进行归一化
    weight_to1 = []
    for eve in weight:
        temp = []
        all = sum(eve)
        for e in eve:
            temp.append(e/all)
        weight_to1.append(temp)

    DBS_clf = DBSCAN(eps=0.4, min_samples=2)
    DBS_clf.fit(weight_to1)
    labels = list(DBS_clf.labels_)
    res = labels_to_original(labels, word)
    print("聚类类簇数：", max(labels)+1)
    print("聚类噪声数：", labels.count(-1))
    print("聚类结果：")
    for eve in res:
        print(eve)



