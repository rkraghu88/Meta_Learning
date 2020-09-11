# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ["TF_XLA_FLAGS"]="--tf_xla_auto_jit=16 --tf_xla_cpu_global_jit"
# os.environ["USE_DAAL4PY_SKLEARN"]="YES"

import MQ_Scheduling_PC_Inst as MQ
import numpy as np
import scipy.stats as stats
import SharedWeights
#import matplotlib
#matplotlib.use("TkAgg")

#plt.switch_backend("TkAgg")

class ThreadRun:
    def __init__(self, lambda_vec,total_users=2,good_users=1,popularity=1.0001,thread_name="default thread", meta_param_len=1, agent_id=0):

        self.ThreadName=thread_name
        self.meta_param_len=meta_param_len
        self.id=agent_id
        self.N_Files = 100
        self.total_users=total_users
        self.good_users=good_users
        self.total_services=100000
        self.analysis_window=10000
        self.service_time=1
        #samples=np.ceil(total_services*service_time*total_lambda)
        self.cache_size=0
        self.x = np.arange(1, self.N_Files+1)
        self.a = popularity
        self.weights = self.x ** (-self.a)
        self.weights /= self.weights.sum()
        self.bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(self.x, self.weights))
        self.lambda_vec=lambda_vec
        # lambda_vec=[.8,2]
        # lambda_vec=[1]
        #lambda_vec=[5]
        self.sojourn_vec=[]
        self.total_lambda=2
        self.FadingMQ=[]

    def runFunc(self):
        for i in range(len(self.lambda_vec)):
            self.total_lambda=self.lambda_vec[i]
            #print(total_lambda)
            self.samples=np.ceil(self.total_services*self.service_time*self.total_lambda)
            seq1 = self.bounded_zipf.rvs(size=int(self.samples+1))
            #inter_arrival_times= stats.expon.rvs(total_lambda, size=int(samples))
            inter_arrival_times = np.cumsum(np.random.exponential(1/(self.total_lambda),int(self.samples)))
            timelines=np.append([0], inter_arrival_times)
            users=stats.randint.rvs(0,self.total_users,size=int(self.samples+1))

            requests=np.array(seq1)
            services=0
            #FadingMQ=MQ.MulticastQueue(requests, timelines, users, service_time, total_users, cache_size)
            self.FadingMQ=MQ.DQNMulticastFadingQueue(requests, timelines, users, self.service_time, self.total_users, self.good_users,self.cache_size,self.total_services, self.ThreadName, self.meta_param_len, self.id)

            SharedWeights.weights=np.append(SharedWeights.weights,self.FadingMQ.DDQNA.meta_model.get_weights())
            self.FadingMQ.DDQNA.update_global_weights()
            ret_val=1
            while(ret_val):
                #print('--')
                #print(FadingMQ.userCaches.cache)
                ret_val=self.FadingMQ.acceptServeRequests()
                #print('--')
                #print(FadingMQ.userCaches.cache)
                #if ret_val==0:
                #    break
                #services=FadingMQ.services
                #print(services)
            #ST=[x for x in FadingMQ.sojournTimes if x.size>0]
            #print(total_lambda,(1-(sum(FadingMQ.userCaches.hit)/sum(FadingMQ.userCaches.requests)))*np.mean(np.concatenate(ST)))
            powSave="pow_vec_%f.txt"%self.total_lambda
            sojSave="soj_vec_%f.txt"%self.total_lambda
            lagSave="beta_vec_%f.txt"%self.total_lambda
            actSave="action_prob_vec_%f.txt"%self.total_lambda
            rewSave="reward_vec_%f.txt"%self.total_lambda
            np.savetxt(powSave,self.FadingMQ.powerVecs)
            np.savetxt(sojSave,self.FadingMQ.sojournTimes)
            np.savetxt(lagSave,self.FadingMQ.DDQNA.penalty_lambda_array)
            np.savetxt(actSave,self.FadingMQ.actionProbVec)
            np.savetxt(rewSave,self.FadingMQ.reward_array)

            print('Final_PVec:',self.FadingMQ.action_prob)
            print('Avg Power:',np.mean(self.FadingMQ.powerVecs[-1000:]))
            print(self.total_lambda,(1-(sum(self.self.FadingMQ.userCaches.hit)/sum(self.FadingMQ.userCaches.requests)))*np.mean(self.FadingMQ.sojournTimes[-np.min([self.analysis_window,self.FadingMQ.sojournTimes.__len__()]).astype(int):]))
            p_hit=(sum(self.FadingMQ.userCaches.hit)/sum(self.FadingMQ.userCaches.requests))
            self.sojourn_vec.append((1-p_hit)*np.mean(self.FadingMQ.sojournTimes[-np.min([self.analysis_window,self.FadingMQ.sojournTimes.__len__()]).astype(int):]))
            # sojourn_vec.append((1-p_hit)*np.mean(FadingMQ.sojournTimes[-np.round(.2*FadingMQ.sojournTimes.__len__()).astype(int):]))
            print(self.FadingMQ.service_vecs, 'Imitation Times:',self.FadingMQ.imit_times)
            #plt.plot(range(FadingMQ.DQNA.reward_array.__len__()),FadingMQ.DQNA.reward_array)
            #plt.show()
            #ST=[x for x in FadingMQ.sojournTimes if x.size>0]
            #print(total_lambda,(1-(sum(FadingMQ.userCaches.hit)/sum(FadingMQ.userCaches.requests)))*np.mean(np.concatenate(ST)))
            #sojourn_vec.append(np.mean(np.concatenate(ST)))
        #plt.plot(lambda_vec,sojourn_vec)
