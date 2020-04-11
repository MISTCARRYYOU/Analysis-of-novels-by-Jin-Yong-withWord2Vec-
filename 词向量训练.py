from gensim.models import word2vec
import timeit


sentences = word2vec.LineSentence('./待训练的分好词的语料库.txt')
t0 = timeit.default_timer()
model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, size=100)
t1 = timeit.default_timer()
print("模型的训练时间为：",t1-t0,"秒")
model.save('./W2Vmodel/model')
