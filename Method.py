import matplotlib.pyplot as plt
import torch
import numpy as np
import random

def getReward(flag, apf, qBefore, q, qNext):
	reward = 0
	# Reward_col
	if flag[0] == 0:
		if flag[1] == 0:
			distance = apf.distanceCost(qNext, apf.obstacle[flag[2],:])
			reward += (distance - apf.Robstacle[flag[2]])/apf.Robstacle[flag[2]] -1
		if flag[1] == 1:
			distance = apf.distanceCost(qNext[0:2], apf.cylinder[flag[2],:])
			reward += (distance - apf.cylinderR[flag[2]])/apf.cylinderR[flag[2]] -1
	else:
		distance1 = apf.distanceCost(qNext, apf.qgoal)
		distance2 = apf.distanceCost(apf.x0, apf.qgoal)
		if distance1 > apf.threshold:
			reward += -distance1/distance2
		else:
			reward += -distance1/distance2 + 2
	return reward

def choose_action(ActorList, s):
	actionList = []
	for i in range(len(ActorList)):
		state = s[i]
		state = torch.as_tensor(state, dtype=torch.float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
		a = ActorList[i](state).cpu().detach().numpy()
		actionList.append(a[0])
		print(a[0])
	return actionList

def drawActionCurve(actionCurveList, obstacleName):
	plt.figure()
	for i in range(actionCurveList.shape[1]):
		array = actionCurveList[:,i]
	plt.plot(np.arange(array.shape[0]), array, linewidth = 2, label = 'Rep%d curve'%i)
	plt.title('Variation diagram of repulsion factor of %s' %obstacleName)
	plt.grid()
	plt.xlabel('time')
	plt.ylabel('value')

def checkCollision(apf, path):
	for i in range(path.shape[0]):
		if apf.checkCollision(path[i,:])[0] == 0:
			return 0
	return 1

def checkPath(apf, path):
	sum = 0
	for i in range(path.shape[0] - 1):
		sum += apf.distanceCost(path[i, :], path[i + 1, :])
	if checkCollision(apf, path) == 1:
		print(sum)
	else:
		print(sum)

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True

def transformAction(actionBefore, actionBound, actionDim):
	actionAfter = []
	for i in range(actionDim):
		action_i = actionBefore[i]
		actionAfter.append((action_i+1)/2*(actionBound[1] - actionBound[0]) + actionBound[0])
	return actionAfter

class Arguments:
	def __init__(self, apf):
		self.obs_dim = 6 * (apf.numberOfSphere + apf.numberOfCylinder)
		self.act_dim = 1 * (apf.numberOfSphere + apf.numberOfCylinder)
		self.act_bound = [0.1, 3]
