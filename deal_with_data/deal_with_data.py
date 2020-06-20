#数据处理
#读取文件
csv_1 = pd.read_csv("file_1.csv")#也可以读txt
test_1 = pd.read_stata("demographic_background.dta")
test_1 =pd.DataFrame(csv_1)#转化成dataframe好操作
#降维
from sklearn.decomposition import PCA
x_reduced=PCA(n_components=3).fit_transform(iris.data)
#时间
#时间从object格式转变为datetime格式
test_1['time2']=pd.to_datetime(test_1['time'],format='%Y/%m/%d  %H:%M:%S.000.')
#用一个日期生成年份列、月份列
one['year']=one['datetime'].apply(lambda x: datetime.datetime.strftime(x,'%Y'))
one['month']=one['datetime'].apply(lambda x: datetime.datetime.strftime(x,'%m'))
#时间可以直接求间隔
test_1['time_inv']=test_1['time2']-test_1['time2'].shift(1)
#df的数据透视功能！超好用 用完记得取消索引
#在python中使用数据透视表pd.pivot_table来计算每年每月的volume累计
add = pd.pivot_table(onetest,index=["year","month"],values=["volume"],aggfunc=np.sum)
#df.pivot在画热力图之前很好用
adduse=add.pivot(index='year',columns='month',values='volume')
#筛选数据
#筛选出某列满足某个条件的df
GPS0 = test_1[test_1['GPSspeed'] == 0.0]
#data[['w','z']]  #选择表格中的'w'、'z'列
test_2 = test_1[['ba001','ba002_1','rgender']]
#看某列的取值分布
test_1['time_inv'].value_counts()

#df连接
df = pd.concat([df1,df2],axis=1)#axis1是横，0是竖

#索引相关
test_4 = test_3.set_index('datetime')
#日期格式设为datetime格式之后神奇的可以如下筛选数据
test_5 = test_4['2008-7':'2008-9']
#释放索引
add=add.reset_index()



#根据已有的某列的取值生成新列
test_5["color"] = test_5["trade_code"].map({'000676.SZ':'lightcoral','000551.SZ':'lightseagreen','000671.SZ':'darkgrey'})





#模型做好后评估参数
#残差平方和
np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
#F统计量 
from scipy.stats import f_oneway  
f,p = f_oneway(clf.predict(X),Y) 

