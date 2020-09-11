import threading as th
import numpy as np


global variab
variab = np.array([1,2])
def thread_function(name,i):
    for j in range(10):
        variab[i]=variab[i]+1
        print('\n',name,i,variab)

t1=th.Thread(group=None,target=thread_function,args=('TH1',0),name='TH1')
t2=th.Thread(group=None,target=thread_function,args=('TH2',1),name='TH2')

t1.start()
t2.start()