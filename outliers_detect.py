

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest

from pyod.models.abod import ABOD

class iforest:

    #사용하는 모델
    #contamination 설정 가능하게 할 것   
    def __init__(self,contamination='auto'):
        self.contamination=contamination
        self.model=IsolationForest(random_state=42,
                                   contamination=self.contamination)
  
    #index값 반환
    #list 형으로 반환
    def iforest_index(self,df):
        self.model.fit(df)
        y_pred=self.model.predict(df)
        df=df.assign(y=y_pred)
        y_out=df.loc[df['y']==-1].index
        y_out=list(y_out)
        return y_out
    
    #이상치 스코어 값 반환
    #npdarray 값으로 반환
    def iforest_score(self,df):
        return self.model.score_samples(df)

