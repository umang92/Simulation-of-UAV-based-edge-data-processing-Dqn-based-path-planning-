# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 16:29:07 2018

@author: wansh
"""
import os

import numpy as np
import gmap as gp
from center_dqn import Center_DQN
from uav import UAV_agent
from sensor import sensor_agent
import matplotlib.pyplot as plt
import time

from gmap import j_region
from gmap import W_wait,find_pos
def map_feature_new(datarate, UAVlist, E_wait, size_fx, size_fy):  # return 600 400 2 feature
    size_f = 84
    size_h = int(size_f / 2)
    sight = 3
    feature_map = np.zeros([size_fx, size_fy, 1])
    num_uav = len(UAVlist)
    for a in range(size_fx):
        for b in range(size_fy):
            # if this is a uav position
            for c in range(num_uav):
                cur_uav = UAVlist[c]
                cur_uav_ps = cur_uav.position
                if int(cur_uav_ps[0]) == a and int(cur_uav_ps[1]) == b:
                    # found a UAV
                    inrange = []
                    position = np.zeros([size_f, size_f, 2])
                    feature = np.zeros([size_f, size_f, 1])

                    for i in range(num_uav):  # find neighbor UAVs
                        ps = UAVlist[i].position
                        No = UAVlist[i].No
                        if No == cur_uav.No:
                            inrange.append(No)
                            continue
                        if ps[0] >= cur_uav_ps[0] - (size_h - 1) * sight - cur_uav.r and ps[0] <= cur_uav_ps[0] + size_h * sight + cur_uav.r and ps[1] >= cur_uav_ps[1] - (size_h - 1) * sight - cur_uav.r and ps[1] <= cur_uav_ps[1] + size_h * sight + cur_uav.r:
                            inrange.append(No)

                    for i in range(size_f):  # define positions of each points in the feature
                        position[:, i, 0] = cur_uav_ps[0] - (size_h - 1) * sight + sight * i
                        position[i, :, 1] = cur_uav_ps[1] + (size_h - 1) * sight - sight * i

                    for i in range(size_f):
                        for j in range(size_f):
                            if position[i, j, 0] < 0 or position[i, j, 0] > cur_uav.region_ifo[-1]['width'] or position[i, j, 1] < 0 or position[i, j, 1] > cur_uav.region_ifo[-1]['hight']:
                                feature[i, j, 0] = 0
                                continue
                            r_no = j_region([position[i, j, 0], position[i, j, 1]],
                                            cur_uav.region_ifo)  # the region No with the current point in
                            pos = find_pos(position[i, j, :])
                            feature[i, j, 0] = datarate[r_no] * E_wait[pos[1], pos[0]]
                            for k in range(len(inrange)):
                                d = np.linalg.norm(np.array([position[i, j, 0] - UAVlist[inrange[k]].position[0],
                                                             position[i, j, 1] - UAVlist[inrange[k]].position[1]]))
                                if d <= cur_uav.r:
                                    if inrange[k] == cur_uav.No:
                                        continue
                                    else:
                                        if feature[i, j, 0] > 0:
                                            feature[i, j, 0] = 0
                                        feature[i, j, 0] = feature[i, j, 0] - 8000

                    # copy computed feature to actual feature map
                    for i in range(size_f):
                        for j in range(size_f):
                            if int(cur_uav_ps[0]) - size_h + i < 0 or int(cur_uav_ps[0]) - size_h + i > size_fx-1 or int(cur_uav_ps[1]) - size_h + j < 0 or int(cur_uav_ps[1]) - size_h + j > size_fy-1:
                                #print(f"buffer overrun {int(cur_uav_ps[0]) - size_h + i}  {int(cur_uav_ps[1]) - size_h + j}")
                                continue
                            feature_map[int(cur_uav_ps[0]) - size_h + i, int(cur_uav_ps[1]) - size_h + j, 0] = max(
                                feature_map[int(cur_uav_ps[0]) - size_h + i, int(cur_uav_ps[1]) - size_h + j, 0],
                                feature[i, j, 0])
                    break

    feature_map = feature_map / 100
    return feature_map.copy()


Ed=5000 #num episodes
num_slots=500                           #total slot
update_target_slot=num_slots/20
ep0=0.97
batch_size=12                 #training samples per batch
pl_step=5                    #How many steps will The system plan the next destination
T=300                          #How many steps will the epslon be reset and the trained weights will be stored
com_r=60
num1=5
num2=4
mapx=300
mapy=300
region=gp.genmap(mapx,mapy,num1,num2)
E_wait=np.ones([mapx+1,mapy+1])
P_cen=np.array([50,50])
t_bandwidth=2e6
N0=2e-20
f_max=2e9    #the max cal frequency of UAV
k=1e-26
cal_L=3000
slot=0.5
num_UAV=6
omeg=1/num_UAV
num_sensor=100
p_max=5
alfmin=1e-3
num_region=num1*num2
C=2e3
v=8
V=10e9
v1=v*np.sin(np.pi/4)
region_obstacle=gp.gen_obs(num_region)
region_rate=np.zeros([num_region])
averate=np.random.uniform(280,300,[num_region])
p_sensor=gp.position_sensor(region,num_sensor)
vlist=[[0,0],[v,0],[v1,v1],[0,v],[-v1,v1],[-v,0],[-v1,-v1],[0,-v],[v1,-v1]]
g0=1e-4
d0=1
the=4
OUT=np.zeros([num_UAV])
reward=np.zeros([num_UAV])
reset_p_T=800

#jud=70000
gammalist=[0,0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9]
Mentrd=np.zeros([Ed, num_slots, num_UAV])

#generate UAV agent
UAVlist=[]
for i in range(num_UAV):
    UAVlist.append(UAV_agent(i,com_r,region_obstacle,region,omeg,slot,t_bandwidth,cal_L,k,f_max,p_max))
    
#generate sensor agent
sensorlist=[]
for i in range(num_sensor):
    sensorlist.append(sensor_agent([p_sensor['W'][i],p_sensor['H'][i]],C,region,averate,slot))


Center=Center_DQN((mapx,mapy,1),num_UAV*9,num_UAV,batch_size)
#Center.load("./save/center-dqn.h5")
prebuf=np.zeros([num_UAV])
data=np.zeros([num_UAV])
#pre_data=np.zeros([num_UAV])

#define record data buf
cover=np.zeros([Ed, num_slots])
#init plt
plt.close()  #clf() # 清图  cla() # 清坐标轴 close() # 关窗口
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plt.xlim((0,600))
plt.ylim((0,400))
plt.grid(True) #添加网格
plt.ion()  #interactive mode on
X=np.zeros([num_UAV])
Y=np.zeros([num_UAV])
fresh_rdw = np.zeros([num_UAV])
fg=1

start=time.time()

def reset():
    E_wait = np.ones([mapx + 1, mapy + 1])
    P_cen = np.array([50, 50])
    for i in range(num_sensor):
        sensorlist.append(sensor_agent([p_sensor['W'][i], p_sensor['H'][i]], C, region, averate, slot))
    for i in range(num_UAV):
        UAVlist.append(UAV_agent(i, com_r, region_obstacle, region, omeg, slot, t_bandwidth, cal_L, k, f_max, p_max))
    prebuf=np.zeros([num_UAV])
    data=np.zeros([num_UAV])
    OUT = np.zeros([num_UAV])
    reward = np.zeros([num_UAV])

print(os.getcwd())
for t in range(Ed):
    reset()
    gp.gen_datarate(averate,region_rate)
    print(f"start episode {t} time elapsed: {time.time()-start}")
    state=None
    for slot in range(num_slots):
        print(f"start slot {slot} of episode {t} time elapsed: {time.time() - start}")
        if slot%pl_step==0:
            aft_feature=[]
            act_note=[]
            pre_feature=map_feature_new(region_rate,UAVlist,E_wait, mapx, mapy)    #record former feature
            act_note=Center.act(pre_feature,fg)          # get the action V

        for i in range(num_UAV):
            OUT[i]=UAVlist[i].fresh_position(vlist[act_note[i]],region_obstacle)     #execute the action
            UAVlist[i].cal_hight()
            X[i]=UAVlist[i].position[0]
            Y[i]=UAVlist[i].position[1]
            UAVlist[i].fresh_buf()
            prebuf[i]=UAVlist[i].data_buf   #the buf after fresh by server

        prevrdw = sum(sum(E_wait))
        gp.list_gama(g0,d0,the,UAVlist,P_cen)

        for i in range(num_sensor):          #fresh buf send data to UAV
            sensorlist[i].data_rate=region_rate[sensorlist[i].rNo]
            sensorlist[i].fresh_buf(UAVlist)
            cover[t, slot]=cover[t, slot]+sensorlist[i].wait
        cover[t, slot]=cover[t, slot]/num_sensor
        print(f"cover : {cover[t, slot]}")
        E_wait = gp.W_wait(mapx, mapy, sensorlist)
        fresh_rdw = sum(sum(E_wait))
        for i in range(num_UAV):
            reward[i] = reward[i] + prevrdw - fresh_rdw
            #reward[i] = reward[i]+UAVlist[i].data_buf-prebuf[i]
            Mentrd[t, slot, i] = reward[i]
            print(f'reward of UAV {i} is {reward[i]} ')
        if state is not None:
            Center.remember(state, act_note, reward, pre_feature, t)
        if slot%pl_step==0:
            state = pre_feature
            if slot >= batch_size:
                Center.replay(batch_size, t, slot)
            for i in range(num_UAV):        #calculate the reward : need the modify
                reward[i]=0
        if slot%update_target_slot==0:
            Center.update_target_model()
            Center.save("./save/center-dqn.h5")
            np.save("record_rd3",Mentrd)

    Center.epsilon=ep0
    Center.save("./save/center-dqn.h5")
    np.save("record_rd3", Mentrd)
    np.save("cover_hungry_10", cover)