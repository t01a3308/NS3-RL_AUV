import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import os

class Painter:
    def __init__(self, load_csv, load_dir=None):
        if not load_csv:
            self.data = pd.DataFrame(columns=['episode reward','episode', 'Method'])
        else:
            self.load_dir = load_dir
            if os.path.exists(self.load_dir):
                self.data = pd.read_csv(self.load_dir).iloc[:,1:]
            else:
                self.data = pd.DataFrame(columns=['episode reward', 'episode', 'Method'])
        self.xlabel = None
        self.ylabel = None
        self.title = None
        self.hue_order = None

    def setXlabel(self,label): self.xlabel = label

    def setYlabel(self, label): self.ylabel = label

    def setTitle(self, label): self.title = label

    def setHueOrder(self,order):
        self.hue_order = order

    def addData(self, dataSeries, method, x=None, smooth = True):
        if smooth:
            dataSeries = self.smooth(dataSeries)
        size = len(dataSeries)
        if x is not None:
            if len(x) != size:
                return
        for i in range(size):
            if x is not None:
                dataToAppend = {'episode reward':dataSeries[i],'episode':x[i],'Method':method}
            else:
                dataToAppend = {'episode reward':dataSeries[i],'episode':i+1,'Method':method}
            self.data = self.data.append(dataToAppend,ignore_index = True)

    def drawFigure(self,style="darkgrid"):
        """
        style: darkgrid, whitegrid, dark, white, ticks
        """
        sns.set_theme(style=style)
        sns.set_style(rc={"linewidth": 1})
        sns.relplot(data = self.data, kind = "line", x = "episode", y = "episode reward",
                    hue= "Method", hue_order=None)
        plt.title(self.title,fontsize = 12)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.show()

    def saveData(self, save_dir):
        self.data.to_csv(save_dir)

    def addCsv(self, add_load_dir):
        add_csv = pd.read_csv(add_load_dir).iloc[:,1:]
        self.data = pd.concat([self.data, add_csv],axis=0,ignore_index=True)

    def deleteData(self,delete_data_name):
        self.data = self.data[~self.data['Method'].isin([delete_data_name])]

    def smoothData(self, smooth_method_name,N):
        begin_index = -1
        mode = -1
        for i in range(len(self.data)):
            if self.data.iloc[i]['Method'] == smooth_method_name and mode == -1:
                begin_index = i
                mode = 1
                continue
            if mode == 1 and self.data.iloc[i]['episode'] == 1:
                self.data.iloc[begin_index:i,0] = self.smooth(
                    self.data.iloc[begin_index:i,0],N = N
                )
                begin_index = -1
                mode = -1
                if self.data.iloc[i]['Method'] == smooth_method_name:
                    begin_index = i
                    mode = 1
            if mode == 1 and i == len(self.data) - 1:
                self.data.iloc[begin_index:,0]= self.smooth(
                    self.data.iloc[begin_index:,0], N=N
                )
    @staticmethod
    def smooth(data,N=5):
        n = (N - 1) // 2
        res = np.zeros(len(data))
        for i in range(len(data)):
            if i <= n - 1:
                res[i] = sum(data[0:2 * i+1]) / (2 * i + 1)
            elif i < len(data) - n:
                res[i] = sum(data[i - n:i + n +1]) / (2 * n + 1)
            else:
                temp = len(data) - i
                res[i] = sum(data[-temp * 2 + 1:]) / (2 * temp - 1)
        return res

if __name__ == "__main__":
    painter = Painter(load_csv=True, load_dir='./figure1.csv')
    painter.drawFigure(style="whitegrid")