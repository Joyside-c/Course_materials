#画图
#分面散点图
grid = sns.FacetGrid(df, col='target', hue='x2', palette='seismic', height=4)
grid.map(plt.scatter, "x3", "x2", alpha=0.5)


#可以直接看一个df里各变量的相关程度
pd.scatter_matrix(test_6,figsize =(10,7))
#更好看的df各变量相关程度，变量什么类型都可以
import seaborn as sns
correlations = test_1.corr() 
correction=abs(correlations)# 取绝对值，只看相关程度 ，不关心正相关还是负相关
# plot correlation matrix 
ax = sns.heatmap(correction,cmap=plt.cm.Greys, linewidths=0.05,vmax=1, vmin=0 ,annot=True,annot_kws={'size':6,'weight':'bold'})

#散点图
plt.scatter(df['x1'],df['x2'],alpha = 0.3)
#饼图
plt.figure(figsize=(6,6))
label = ['Snake','Rabbit','Dragon','Tiger','Horse','Ox','Rooster','Goat','Rat','Monkey','Pig','Dog']
plt.pie(test_2['ba001'].value_counts(),labels=label,autopct = '%3.1f%%',labeldistance=1.2)
plt.legend(loc=4, bbox_to_anchor=(1.5,0.2))
plt.title('animal ratio')
#bar
plt.figure(figsize=(10,6))
#设置x轴柱子的个数
x= np.arange(len(date))+1 #课程品类数量已知为14，也可以用len(ppv3.index)
#设置y轴的数值，需将numbers列的数据先转化为数列，再转化为矩阵格式
y=np.array(date)
xticks1=list(date.index) #构造不同课程类目的数列
#画出柱状图
plt.bar(x,y,width = 0.5,align='center',color = 'black',alpha=0.8)
#设置x轴的刻度，将构建的xticks代入
plt.xticks(x,xticks1,size='small',rotation=30)
#x、y轴标签与图形标题
plt.xlabel('animal')
plt.ylabel('count')
plt.title('animal count')
#设置数字标签**
for a,b in zip(x,y):
    plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=11)
#标注    
plt.annotate("60%的销售量被前55种面膜占有", (55,615322),xytext=(155,615322),arrowprops=dict(arrowstyle='->')) 
#设置y轴的范围
plt.ylim(0,2000)
plt.show()
#直方图
plt.hist(np.array(test_2['ba002_1']),color = 'coral')
plt.title('birth year')
plt.xlabel('year')
plt.ylabel('count')
#sns的直方图（better）
plt.figure(figsize = (5,3))
sns.set_palette("hls") 
sns.distplot(chg,color="r",bins=80,kde=True,hist=True,rug=True)
#sns的多面直方图
sns.set( palette="dark", color_codes=True)  
rs = np.random.RandomState(10)  
d = rs.normal(size=100)  
f, axes = plt.subplots(2, 2, figsize=(7, 4), sharex=True)  
sns.distplot(chg, kde=False, color="b", ax=axes[0, 0])  
sns.distplot(chg, hist=False, rug=True, color="r", ax=axes[0, 1])  
sns.distplot(chg,hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])  
sns.distplot(chg,color="m", ax=axes[1, 1])  
plt.xlim(-0.7,0.7)
plt.show() 
#分面图
import seaborn as sns
#FacetGrid 是一个绘制多个图表（以网格形式显示）的接口。
g2 = sns.FacetGrid(test_2,  col="rgender")
g2 = (g2.map(plt.scatter,"ba002_1" ,"ba001").add_legend())
#箱线图
sns.catplot(x='ba001',y='ba002_1',hue='rgender',kind='box',
           data=test_2,aspect=2) 
#厉害的动态图,缺点是对数据的处理能力不强，需要生成好要求的list形式然后放进去
from IPython.display import SVG
plt.figure(figsize=(5,5))
radar_chart = pygal.Radar()
radar_chart.title = 'rador'
radar_chart.x_labels = ['open','high','low','close','turn']
radar_chart.add('2008-01-02', [6.90,7.18,6.88,7.14,2.1676])
radar_chart.add('2008-01-03', [7.14,7.43,7.09,7.38,3.0858])
radar_chart.add('2008-01-04', [7.40,7.44,7.22,7.30,1.7568])
display(SVG(radar_chart.render()))

#保存矢量图
plt.savefig("fig.eps")
