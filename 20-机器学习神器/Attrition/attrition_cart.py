import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import warnings
warnings.filterwarnings('ignore')  # 关闭所有警告信息

# 数据加载
train=pd.read_csv('train.csv',index_col=0)
test=pd.read_csv('test.csv',index_col=0)

# 数据探索
print(train['Attrition'].value_counts())

# 处理Attrition字段, 可以使用map 进行自定义，也可以使用LabelEncoder进行自动的标签编码
train['Attrition']=train['Attrition'].map(lambda x:1 if x=='Yes' else 0)
print(train['Attrition'].value_counts())

from sklearn.preprocessing import LabelEncoder
# 查看数据是否有空值
print(train.isnull().sum())
# 如果方差为0, 没有意义
print(train['StandardHours'].value_counts())

# 去掉没用的列 员工号码，标准工时（=80）
train = train.drop(['EmployeeNumber', 'StandardHours'], axis=1)
test = test.drop(['EmployeeNumber', 'StandardHours'], axis=1)
print(train.info())

# 对于分类特征进行特征值编码
attr=['BusinessTravel','Department','Education','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime']
lbe_list=[]
for feature in attr:
    lbe=LabelEncoder()
    train[feature]=lbe.fit_transform(train[feature])
    test[feature]=lbe.transform(test[feature])
    lbe_list.append(lbe)
train.to_csv('train_label_encoder.csv')

# 建模环节，CART决策树分类模型
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib  # 用于模型保存
# 修复matplotlib中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
# 数据集进行切分，20%用于测试
X_train, X_valid, y_train, y_valid = train_test_split(train.drop('Attrition',axis=1), train['Attrition'], test_size=0.2, random_state=2025)

# CART决策树分类模型，最大深度为4
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    random_state=42
)
# 模型训练
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, 'cart_model.pkl')  # 保存为cart_model.pkl
print('决策树模型已保存为 cart_model.pkl')

# 打印决策树规则
print('CART决策树规则：')
feature_names = X_train.columns
rules = export_text(model, feature_names=list(feature_names))
print(rules)

# 决策树可视化
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=feature_names, class_names=['No', 'Yes'], filled=True, rounded=True, fontsize=10)
plt.title('CART决策树（最大深度4）')
plt.show()

# 预测与输出
predict = model.predict(test)
print('标签Label：')
print(predict)
test['Attrition']=predict
print(test['Attrition'])
test[['Attrition']].to_csv('submit_cart.csv')
print('submit_cart.csv saved')
# 转化为二分类输出
test['Attrition']=test['Attrition'].map(lambda x:1 if x>=0.5 else 0)
#test[['Attrition']].to_csv('submit_cart.csv') 