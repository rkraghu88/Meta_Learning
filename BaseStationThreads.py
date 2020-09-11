import threading as th
import numpy as np
import SchedPCFadingMulticasting as BaseStations
import SharedWeights
import tensorflow as tf

lambda_vec_bs=np.array([0.1,0.2])
#BaseStations.ThreadRun(lambda_vec_bs,total_users=10,good_users=5,popularity=1.0001,thread_name="BS_Thread").runFunc()
BS=np.array([])
TH=np.array([])
SharedWeights.weights=np.array([])

for i in np.arange(len(lambda_vec_bs)):
    BS=np.append(BS,[BaseStations.ThreadRun([lambda_vec_bs[i]],total_users=2,good_users=1,popularity=1.0001,thread_name="BS_Thread_%d"%i,meta_param_len=lambda_vec_bs.__len__(), agent_id=i.astype(int))])
    TH=np.append(TH,[th.Thread(group=None,target=BS[i].runFunc,name=BS[i].ThreadName)])
    TH[i].start()
    print('Thread',i,'Started')
