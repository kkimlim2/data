

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

class if_outliers:
    
    #사용하는 모델
    #contamination 설정 가능하게 할 것   
    def __init__(self,contamination='auto'):
        self.contamination=contamination
        self.model=IsolationForest(random_state=42,
                                   contamination=self.contamination)


  
    #index값 반환
    #list 형으로 반환
    def index(self,df):
        self.cols=list(df.columns)
        for i,value in enumerate(self.cols):
            if df[value].dtype != np.number:
                del self.cols[i]
        self.df=pd.DataFrame(df[self.cols].values)
        self.model.fit(self.df)
        y_pred=self.model.predict(self.df)
        self.df=self.df.assign(y=y_pred)
        y_out=self.df.loc[self.df['y']==-1].index
        self.y_out=list(y_out)
        return self.y_out
    

    
    #이상치 스코어 값 반환
    #npdarray 값으로 반환
    def score(self,df):
        self.df=self.df.drop(self.df[['y']],axis=1)
        self.x=self.model.score_samples(self.df)
        return self.x

    

    #이상치 비율 값 반환
    def rate(self,df):
        self.ratio=len(self.y_out)/self.df.shape[0]
        return self.ratio

    

    #이상치 스코어 그래프 그리기
    def visualize(self,df):        
        fig=plt.figure(figsize=(30,30))
        plt.plot(self.df.index,self.x)
        plt.scatter(self.y_out,self.x[[i for i in self.y_out]],
                    edgecolor="k",color="red")
        plt.show()





    
   



        
