import jieba


# 先统计预料库中人名出现的频率，并写入人名词频统计.txt
def readtxt(path):
    with open(path, "r", encoding="utf-8") as f:
        cor = [eve.strip("\n") for eve in f]
    return cor


names = readtxt("文本文档/人名专有词词典.txt")
b = readtxt("文本文档/待训练的分好词的语料库.txt")
all = []
for eve in b:
    all += eve.split()

name_with_frequence = []
names = list(set(names))
print("开始统计！")
for name in names:
    try:
        name_with_frequence.append((name, all.count(name)))
    except:
        print("word", name, "not found!")
sorone = sorted(name_with_frequence, key=lambda tuple_: tuple_[1], reverse=True)
with open("文本文档/全小说总的人名词频统计.txt", "w", encoding="utf-8") as f:
    for eve in sorone:
        print(eve[0], " ", str(eve[1]), file=f)

# 按照各个小说，统计其前10个出现频率最高的人名的频率，依次写入 全小说各自的人名词频统计.txt
names_14 = []
split_point = [0,85,138,167,279,438,455,522,789,921,1047,1108,1143,1406,1421]
for i in range(len(split_point)-1):
    names_14.append(names[split_point[i]:split_point[i+1]])
frequence_14 = [[] for i in range(14)]
for i in range(14):
    for j in range(len(names_14[i])):
        frequence_14[i].append((names_14[i][j], all.count(names_14[i][j])))
# 进行排序
write = []
for i in range(14):
    frequence_14[i] = sorted(frequence_14[i], key=lambda tuple_: tuple_[1], reverse=True)
    write.append([eve[0] for eve in frequence_14[i]])
with open("文本文档/全小说各自的人名词频统计.txt", "w", encoding="utf-8") as f:
    for eve in write:
        print(" ".join(eve), file=f)
# 14部小说的人名分割点(左闭右开)：[0,85,167,279,438,455,522,789,921,1047,1108,1143,1406]


# 开始对文本进行分词，并写入 待训练的分好词的语料库.txt
jieba.load_userdict("人名专有词词典.txt")
# 由于小说中奇怪的人名较多，需要利用专有词；但是由于w2v的特点，不需要使用停用词，标点也不需要去除。
with open(r"文本文档/预处理后的文本.txt", "r", encoding="utf-8") as f:
    sentences = []
    for eve in f:
        if eve == "":
            continue
        cutword = jieba.lcut(eve.strip("\n"))
        clean_cutword = []
        for ev in cutword:
            if ev == "":
                continue
            clean_cutword.append(ev)
        sentences.append(" ".join(clean_cutword))
with open(r"文本文档/待训练的分好词的语料库.txt", "w", encoding="utf-8") as f:
    for eve in sentences:
        print(eve, file=f)

