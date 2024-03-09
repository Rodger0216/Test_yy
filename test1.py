import pandas as pd
df=pd.read_excel(r"D:\YYfiles\000PhD\2024WDSA-CCWI Joint Conference\Urban Water Team of DUT_Materials-W1\W1\InflowData_1.xlsx",index_col=0)# index_col表示指定索引的列
df.index.dtype# 查看索引是否为datatimeindex
type(df)# 查看df的数据类型


df=df.resample('M').mean()# 计算每个月的平均值
# “M”代表月，还可以换成年，星期等，需要首先将索引换成datetimeindex.
import matplotlib.pyplot as plt
df.plot(figsize=(12,8))# 检查数据平稳性,df后面直接加plot.
df["差分"]=df['流量（m3/s)'].diff(1)
df["差分"].plot(figsize=(12,8))# 查看经过一阶差分处理之后数据的平稳性。
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
a=plot_acf(df["差分"].dropna())# 绘出差分数据（记住是差分数据）acf和pacf图，并找出p，q值。
b=plot_pacf(df["差分"].dropna())# 需要dropna()
import statsmodels.api as sm
model = sm.tsa.arima.ARIMA(df["流量（m3/s)"], order=(11,1,2))# 建立arima模型
# order是（p,d,q），这里输入的是原数据，而不是差分数据，也不需要dropna()。
res=model.fit()# 训练
pred=res.predict('20211231','20221231')# 进行预测，就算是已知数据通过预测之后还是和已知数据不一样。
print(pred)# 输出预测