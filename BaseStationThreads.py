import threading as th
import numpy as np
import SchedPCFadingMulticasting as BaseStations
import SharedWeights
from collections import deque
import tensorflow as tf

lambda_vec_bs=np.array([0.1,0.2])
#BaseStations.ThreadRun(lambda_vec_bs,total_users=10,good_users=5,popularity=1.0001,thread_name="BS_Thread").runFunc()
BS=np.array([])
TH=np.array([])
# meta_parameter=np.array([1.0])
# for k in range(len(lambda_vec_bs)-1):
#     meta_parameter=np.append(meta_parameter,0.0)

meta_parameter=np.identity(len(lambda_vec_bs))
SharedWeights.weights=deque()
# SharedWeights.weights=np.array([])
SharedWeights.weight_size=0
for i in np.arange(len(lambda_vec_bs)):
    BS=np.append(BS,[BaseStations.ThreadRun([lambda_vec_bs[i]],total_users=2,good_users=1,popularity=1.0001,thread_name="BS_Thread_%d"%i,meta_parameter=meta_parameter[i],meta_param_len=lambda_vec_bs.__len__(), agent_id=i.astype(int))])
    TH=np.append(TH,[th.Thread(group=None,target=BS[i].runFunc,name=BS[i].ThreadName)])
    TH[i].start()
    print('Thread',i,'Started')
