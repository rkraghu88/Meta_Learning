import threading as th
import numpy as np
import SchedPCFadingMulticasting as BaseStations

lambda_vec_bs=np.array([0.1,0.5,0.8,0.2])
BS=np.array([])
TH=np.array([])
for i in range(len(lambda_vec_bs)):
    BS=np.append(BS,[BaseStations.ThreadRun([lambda_vec_bs[i]],total_users=10,good_users=5,popularity=1.0001,thread_name="BS_Thread_%d"%i)])
    TH=np.append(TH,[th.Thread(group=None,target=BS[i].runFunc,name=BS[i].ThreadName)])
    TH[i].start()
    print('Thread',i,'Started')


