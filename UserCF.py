# movieslens数据集实现UserCF算法

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math

# 使用pandas导入数据
moviesDF = pd.read_csv("./ml-latest-small/movies.csv",index_col=None)
ratingsDF = pd.read_csv("./ml-latest-small/ratings.csv",index_col=None)
moviesDF.head()

# 创建训练集和测试集（最后也没用到测试集）
trainRatingsDF, testRatingsDF = train_test_split(ratingsDF, test_size=0.2)

# 将训练数据使用pandas透视表生成用户对电影的评分矩阵
trainRatingsPivotDF = pd.pivot_table(trainRatingsDF[['userId','movieId','rating']],columns=['movieId'],index=['userId'],values='rating',fill_value=0)
# 为用户和电影添加索引
moviesMap = dict(enumerate(list(trainRatingsPivotDF.columns)))
usersMap = dict(enumerate(list(trainRatingsPivotDF.index)))
# ratingValues存放用户对电影的评分矩阵 (610, 8899)
ratingValues = trainRatingsPivotDF.values.tolist()

# 使用余弦相似度求用户间的相似度
def calCosineSimilarity(list1,list2):
    res = 0
    denominator1 = 0
    denominator2 = 0
    for (val1,val2) in zip(list1,list2):
        res += (val1 * val2)
        denominator1 += val1 ** 2
        denominator2 += val2 ** 2
    return res / (math.sqrt(denominator1 * denominator2))

# 计算用户间的相似度存入用户相似度矩阵，矩阵对称，对角线为0
userSimMatrix = np.zeros((len(ratingValues),len(ratingValues)),dtype=np.float32)
for i in range(len(ratingValues)-1):
    for j in range(i+1,len(ratingValues)):
        userSimMatrix[i,j] = calCosineSimilarity(ratingValues[i],ratingValues[j])
        userSimMatrix[j,i] = userSimMatrix[i,j]

# 选出与每个用户最相近的十个用户存入字典中
userMostSimDict = dict() # 与每个用户最相近的K个用户
for i in range(len(ratingValues)):
    userMostSimDict[i] = sorted(enumerate(list(userSimMatrix[i])),key = lambda x:x[1],reverse=True)[:10]
    # enumerate为userSimMatrix的数据添加索引，sorted中的key = lambda x:x[1]表示对待排序对象中的第x[1]维（也就是第二维）进行排序
    
# 得到用户对每个没有观看过的电影的兴趣分
userRecommendValues = np.zeros((len(ratingValues),len(ratingValues[0])),dtype=np.float32)
# userRecommendValues.shape (610, 8899)
for i in range(len(ratingValues)):
    for j in range(len(ratingValues[i])):  # len(ratingValues[0])= 8899
        if ratingValues[i][j] == 0:  # 如果用户对电影j没操作
            val = 0
            for (user,sim) in userMostSimDict[i]:  # 如果与用户i最相似的那些用户对电影j有操作
                val += (ratingValues[user][j] * sim)  # 使用用户间的相似度与相似用户对电影评分的乘积之和作为用户对电影j的兴趣分
                userRecommendValues[i,j] = val 

# 为每个用户推荐10部电影
userRecommendDict = dict()
for i in range(len(ratingValues)):
    userRecommendDict[i] = sorted(enumerate(list(userRecommendValues[i])),key = lambda x:x[1],reverse=True)[:10]

# 将索引的用户id和电影id转换为真正的用户id和电影id
userRecommendList = []
for key,value in userRecommendDict.items():
    user = usersMap[key]
    for (movieId,val) in value:
        userRecommendList.append([user,moviesMap[movieId]])

# 我们将推荐结果的电影id转换成对应的电影名，并打印结果：
recommendDF = pd.DataFrame(userRecommendList,columns=['userId','movieId'])
recommendDF = pd.merge(recommendDF,moviesDF[['movieId','title']],on='movieId',how='inner')  #pandas的合并
recommendDF
