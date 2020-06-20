#分层抽样，以分级作groupby
gbr = data_u.groupby("grade")
a=gbr.agg(np.size)
#用按比例的分层抽样，需要一个抽样比例的typicalFracDict,用前两列计数做
a["frac"] = a.id/sum(a.id)
a=a.reset_index()
b = a[["grade","frac"]]
fracdict = b.set_index('grade').T.to_dict("record")
typicalFracDict=fracdict[0]


#对某列分组
# 划定分组的分界值
X_dot=[0]
for i in range(10000,110000,10000):
    X_dot.append(i)
X_dot.append(150000)
for i in range(200000,500000,100000):
    X_dot.append(i)
X_dot.append(max(data_u.annual_inc))

# 数据放入组中
X_group = pd.cut(data_u.annual_inc,bins=X_dot,right=True,retbins=False,include_lowest=False)

# 计算频数，利用groupby的方法
data=pd.DataFrame(data_u.annual_inc)
data["group"]=X_group
groupValueSize=data.groupby(X_group).size()
pinshulist_T=list(pd.DataFrame(groupValueSize).reset_index().iloc[:,1])

#抽样函数
#按比例分层抽样，将样本量当作参数传入，可调整
samplenum=list(np.trunc(2**np.array(np.arange(5,18,0.2))).astype(np.int32))
def typicalSampling(group, typicalFracDict,n):
    name = group.name
    return group.sample(frac=n/235629)

def get_sample_fc(samplenum):
    pinshulist_sam1=[]
    for i in  samplenum:
        result = list(data_u.groupby('grade', group_keys=False).apply(typicalSampling, typicalFracDict,i).annual_inc)
        pinshulist_sam1.append(result)
    return pinshulist_sam1
    
#简单随机抽样
def get_sample_jd(samplenum):
    pinshulist_sam2=[]
    for i in samplenum:
        result = list(data_u.sample(i).annual_inc)
        pinshulist_sam2.append(result)
    return pinshulist_sam2

#对抽样出来的数据分组求KL(分层)
def get_KL(get_sample_fc):
    pinshulist_fc_all=[]
    for i in get_sample_fc(samplenum):
        X_group = pd.cut(i,bins=X_dot,right=True,retbins=False,include_lowest=False)
        data=pd.DataFrame(i)
        data["group"]=X_group
        groupValueSize=data.groupby(X_group).size()
        pinshulist_fc=list(pd.DataFrame(groupValueSize).reset_index().iloc[:,1])
        pinshulist_fc_all.append(pinshulist_fc)
    #计算交叉熵
    shang_fc_l=[]
    for i in  pinshulist_fc_all:
        shang_fc = scipy.stats.entropy(pinshulist_T, i)
        shang_fc_l.append(shang_fc)    
    return shang_fc_l
#每个样本量下抽取50次(分层)
i=0
#pinshulist_fc_all_meanpre=[]
shang_fc_l_meanpre=[]
while i < 50:
    get_sample_fc(samplenum)
    get_KL(get_sample_fc)
    #pinshulist_fc_all_meanpre.append(pinshulist_fc_all)
    shang_fc_l_meanpre.append(get_KL(get_sample_fc))
    i=i+1


t_0=np.array(shang_fc_l_meanpre)
arr_ini=np.zeros(65)
for i in np.arange(50):
    arr = (arr_ini+t_0[i])/50
#简单随机
def get_KL(get_sample_jd):
    pinshulist_jd_all=[]
    for i in get_sample_jd(samplenum):
        X_group = pd.cut(i,bins=X_dot,right=True,retbins=False,include_lowest=False)
        data=pd.DataFrame(i)
        data["group"]=X_group
        groupValueSize=data.groupby(X_group).size()
        pinshulist_jd=list(pd.DataFrame(groupValueSize).reset_index().iloc[:,1])
        pinshulist_jd_all.append(pinshulist_jd)    
     #计算交叉熵
    shang_jd_l=[]
    for i in  pinshulist_jd_all:
        shang_jd = scipy.stats.entropy(pinshulist_T, i)
        shang_jd_l.append(shang_jd)
    return shang_jd_l
i=0
#pinshulist_jd_all_meanpre=[]
shang_jd_l_meanpre=[]
while i < 50:
    get_sample_fc(samplenum)
    get_KL(get_sample_jd)
    #pinshulist_jd_all_meanpre.append(pinshulist_jd_all)
    shang_jd_l_meanpre.append(get_KL(get_sample_jd))
    i=i+1
j_0=np.array(shang_jd_l_meanpre)
arr_ini=np.zeros(65)
for i in np.arange(50):
    arr1 = (arr_ini+j_0[i])/50