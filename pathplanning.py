import torch
import matplotlib.pyplot as plt
import numpy as np
from ApfAlgorithm import *
from Method import choose_action, checkPath, \
                   drawActionCurve, getReward, \
                   transformAction, Arguments
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def PathPlanning():
    apf = APF()
    args = Arguments(apf)
    centralizedActor = torch.load('TrainedModel/centralizedActor.pkl',map_location=device)
    actionCurveDic = {'sphere':np.array([]), 'cylinder':np.array([])}

    q = apf.x0        
    qBefore = [None, None, None]
    rewardSum = 0

    for i in range(500):
        obsDicq = apf.calculateDynamicState(q)
        obs_sphere, obs_cylinder = obsDicq['sphere'], obsDicq['cylinder']
        obs_mix = obs_sphere + obs_cylinder 
        obs = np.array([])
        for k in range(len(obs_mix)):
            obs = np.hstack((obs, obs_mix[k]))  
        obs = torch.as_tensor(obs, dtype=torch.float, device=device)
        action = centralizedActor(obs).cpu().detach().numpy()
        action = transformAction(action, args.act_bound, args.act_dim)
        action_sphere = action[0:apf.numberOfSphere]
        action_cylinder = action[apf.numberOfSphere:apf.numberOfSphere + apf.numberOfCylinder]
        actionCurveDic['sphere'] = np.append(actionCurveDic['sphere'], action_sphere)
        actionCurveDic['cylinder'] = np.append(actionCurveDic['cylinder'], action_cylinder)

        qNext = apf.getqNext(apf.epsilon0, action_sphere, action_cylinder, q, qBefore)
        qBefore = q

        flag = apf.checkCollision(qNext)
        rewardSum += getReward(flag, apf, qBefore, q, qNext)

        q = qNext
        if apf.distanceCost(q,apf.qgoal) < apf.threshold:
            apf.path = np.vstack((apf.path,apf.qgoal))
            break
    checkPath(apf,apf.path)
    apf.saveCSV()
    print('reward : %f' % rewardSum)
    plt.show()