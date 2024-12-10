import pandas as pd

df = pd.read_csv('../data/정밀가공_품질보증_데이터셋.csv')

ratio = 30
sum = 0

for column in df.columns:
    tmp = df[column]
    print('column name :', column)
    print('[step 1] 변수별 결측 비율')
    print(round(tmp.isnull().sum()/len(tmp)*100, 2))
    print('[step 2] 변수별 결측 비율 30% 초과 여부')
    print(tmp.isnull().sum()/len(tmp)*100 > ratio)
    cmpt_len = tmp.isnull().sum().sum()
    print('[step 3] 전체 데이터셋 결측치 개수')
    print(cmpt_len)
    print(f"결측치 = {cmpt_len}개")
    print(f"완전성 지수 : {(1-cmpt_len/len(df))*100}%")
    print('='*30)
    sum += (1-cmpt_len/len(df))*100