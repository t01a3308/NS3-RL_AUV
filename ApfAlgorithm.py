import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from static_obstacle_environment import Obstacle

class APF:
	def __init__(self):
		env = 'Obstacle1'
		self.obstacle = Obstacle[env].obstacle 
		self.Robstacle = Obstacle[env].Robstacle
		self.cylinder = Obstacle[env].cylinder      
		self.cylinderR = Obstacle[env].cylinderR    
		self.cylinderH  = Obstacle[env].cylinderH 

		self.numberOfSphere = self.obstacle.shape[0]
		self.numberOfCylinder = self.cylinder.shape[0]

		self.qgoal = Obstacle[env].qgoal
		self.x0 = Obstacle[env].x0
		self.stepSize = 0.2
		self.dgoal = 5
		self.r0 = 5
		self.threshold = 0.2

		self.xmax = 10/180 * np.pi

		self.gammax = 10/180 * np.pi

		self.maximumClimbingAngle = 100/180 * np.pi
		self.maximumSubductionAngle = -75/180 * np.pi 

		self.path = self.x0.copy()
		self.path = self.path[np.newaxis, :]

		self.epsilon0 = 0.8
		self.eta0 = 0.5

	def reset(self):
		self.path = self.x0.copy()
		self.path = self.path[np.newaxis, :]

	def calculateDynamicState(self, q):
		dic = {'sphere':[], 'cylinder':[]}
		sAll = self.qgoal - q
		for i in range(self.numberOfSphere):
		    s1 = self.obstacle[i,:] - q
		    dic['sphere'].append(np.hstack((s1,sAll)))
		for i in range(self.numberOfCylinder):
		    s1 = np.hstack((self.cylinder[i,:],q[2])) - q
		    dic['cylinder'].append(np.hstack((s1, sAll)))
		return dic

	def inRepulsionArea(self, q):
		dic = {'sphere':[], 'cylinder':[]}
		for i in range(self.numberOfSphere):
		    if self.distanceCost(q, self.obstacle[i,:]) < self.r0:
		        dic['sphere'].append(i)
		for i in range(self.numberOfCylinder):
		    if self.distanceCost(q[0:2], self.cylinder[i,:]) < self.r0:
		        dic['cylinder'].append(i)
		return dic

	def attraction(self, q, epsilon):
		r = self.distanceCost(q, self.qgoal)
		if r <= self.dgoal:
		    fx = epsilon * (self.qgoal[0] - q[0])
		    fy = epsilon * (self.qgoal[1] - q[1])
		    fz = epsilon * (self.qgoal[2] - q[2])
		else:
		    fx = self.dgoal * epsilon * (self.qgoal[0] - q[0]) / r
		    fy = self.dgoal * epsilon * (self.qgoal[1] - q[1]) / r
		    fz = self.dgoal * epsilon * (self.qgoal[2] - q[2]) / r
		return np.array([fx, fy, fz])

	def repulsion(self, q, eta):
		f0 = np.array([0, 0, 0])
		Rq2qgoal = self.distanceCost(q, self.qgoal)
		for i in range(self.obstacle.shape[0]):
			r = self.distanceCost(q, self.obstacle[i, :])
			if r <= self.r0:
				tempfvec = eta * (1 / r - 1 / self.r0) * Rq2qgoal ** 2 / r ** 2 * self.differential(q, self.obstacle[i, :]) \
				+ eta * (1 / r - 1 / self.r0) ** 2 * Rq2qgoal * self.differential(q, self.qgoal)
				f0 = f0 + tempfvec
			else:
				tempfvec = np.array([0, 0, 0])
				f0 = f0 + tempfvec

			for i in range(self.cylinder.shape[0]):
				r = self.distanceCost(q[0:2], self.cylinder[i, :])
				if r <= self.r0:
					repulsionCenter = np.hstack((self.cylinder[i,:],q[2]))
					tempfvec = eta * (1 / r - 1 / self.r0) * Rq2qgoal ** 2 / r ** 2 * self.differential(q, repulsionCenter) \
					+ eta * (1 / r - 1 / self.r0) ** 2 * Rq2qgoal * self.differential(q, self.qgoal)
					f0 = f0 + tempfvec
				else:
					tempfvec = np.array([0, 0, 0])
					f0 = f0 + tempfvec
			return f0

	def repulsionForOneObstacle(self, q, eta, qobs):
		f0 = np.array([0, 0, 0])
		Rq2qgoal = self.distanceCost(q, self.qgoal)
		r = self.distanceCost(q, qobs)
		if r <= self.r0:
			tempfvec = eta * (1 / r - 1 / self.r0) * Rq2qgoal ** 2 / r ** 2 * self.differential(q, qobs) \
			+ eta * (1 / r - 1 / self.r0) ** 2 * Rq2qgoal * self.differential(q, self.qgoal)
			f0 = f0 + tempfvec
		else:
			tempfvec = np.array([0, 0, 0])
			f0 = f0 + tempfvec
		return f0

	def dynamicRepulsion(self, q):
		f0 = np.array([0, 0, 0])
		Rq2qgoal = self.distanceCost(q, self.qgoal)
		r = self.distanceCost(q, self.dynamicSphereXYZ)
		if r <= self.dynamicSpherer0:
			tempfvec = self.dynamicSphereEta * (1 / r - 1 / self.dynamicSpherer0) * Rq2qgoal ** 2 / r ** 2 * self.differential(q,self.dynamicSphereXYZ) \
			+ self.dynamicSphereEta * (1 / r - 1 / self.dynamicSpherer0) ** 2 * Rq2qgoal * self.differential(q, self.qgoal)
			f0 = f0 + tempfvec
		else:
			tempfvec = np.array([0, 0, 0])
			f0 = f0 + tempfvec
		return f0

	def differential(self, q, other):
		output1 = (q[0] - other[0]) / self.distanceCost(q, other)
		output2 = (q[1] - other[1]) / self.distanceCost(q, other)
		output3 = (q[2] - other[2]) / self.distanceCost(q, other)
		return np.array([output1, output2, output3])

	def getqNext(self, epsilon, eta1List, eta2List, q, qBefore):
		qBefore = np.array(qBefore)
		if qBefore[0] is None:
			unitCompositeForce = self.getUnitCompositeForce(q, eta1List, eta2List, epsilon)
			qNext = q + self.stepSize * unitCompositeForce 
		else:
			unitCompositeForce = self.getUnitCompositeForce(q, eta1List, eta2List, epsilon)
			qNext = q + self.stepSize * unitCompositeForce
			_, _, _, _, qNext = self.kinematicConstrant(q, qBefore, qNext)
			self.path = np.vstack((self.path, qNext))
		return qNext

	def getUnitCompositeForce(self, q, eta1List, eta2List, epsilon):
		Attraction = self.attraction(q, epsilon)  
		Repulsion = np.array([0,0,0])
		for i in range(len(eta1List)): 
			Repulsion = Repulsion + self.repulsionForOneObstacle(q, eta1List[i], self.obstacle[i,:])
		for i in range(len(eta2List)):
			Repulsion = Repulsion + self.repulsionForOneObstacle(q, eta2List[i], np.hstack((self.cylinder[i,:],q[2])))
			compositeForce = Attraction + Repulsion  
			unitCompositeForce = self.getUnitVec(compositeForce)  
		return unitCompositeForce

	def kinematicConstrant(self, q, qBefore, qNext):
		qBefore2q = q - qBefore
		if qBefore2q[0] != 0 or qBefore2q[1] != 0:
			x1 = np.arcsin(np.abs(qBefore2q[1] / np.sqrt(qBefore2q[0] ** 2 + qBefore2q[1] ** 2)))
			gam1 = np.arcsin(qBefore2q[2] / np.sqrt(np.sum(qBefore2q ** 2)))
		else:
			return None, None, None, None, qNext
		q2qNext = qNext - q
		x2 = np.arcsin(np.abs(q2qNext[1] / np.sqrt(q2qNext[0] ** 2 + q2qNext[1] ** 2)))
		gam2 = np.arcsin(q2qNext[2] / np.sqrt(np.sum(q2qNext ** 2)))

		if qBefore2q[0] > 0 and qBefore2q[1] > 0:
			x1 = x1
		if qBefore2q[0] < 0 and qBefore2q[1] > 0:
			x1 = np.pi - x1
		if qBefore2q[0] < 0 and qBefore2q[1] < 0:
			x1 = np.pi + x1
		if qBefore2q[0] > 0 and qBefore2q[1] < 0:
			x1 = 2 * np.pi - x1
		if qBefore2q[0] > 0 and qBefore2q[1] == 0:
			x1 = 0
		if qBefore2q[0] == 0 and qBefore2q[1] > 0:
			x1 = np.pi / 2
		if qBefore2q[0] < 0 and qBefore2q[1] == 0:
			x1 = np.pi
		if qBefore2q[0] == 0 and qBefore2q[1] < 0:
			x1 = np.pi * 3 / 2

		if q2qNext[0] > 0 and q2qNext[1] > 0:
			x2 = x2
		if q2qNext[0] < 0 and q2qNext[1] > 0:
			x2 = np.pi - x2
		if q2qNext[0] < 0 and q2qNext[1] < 0:
			x2 = np.pi + x2
		if q2qNext[0] > 0 and q2qNext[1] < 0:
			x2 = 2 * np.pi - x2
		if q2qNext[0] > 0 and q2qNext[1] == 0:
			x2 = 0
		if q2qNext[0] == 0 and q2qNext[1] > 0:
			x2 = np.pi / 2
		if q2qNext[0] < 0 and q2qNext[1] == 0:
			x2 = np.pi
		if q2qNext[0] == 0 and q2qNext[1] < 0:
			x2 = np.pi * 3 / 2

		deltax1x2 = self.angleVec(q2qNext[0:2], qBefore2q[0:2])
		if deltax1x2 < self.xmax:
			xres = x2
		elif x1 - x2 > 0 and x1 - x2 < np.pi: 
			xres = x1 - self.xmax
		elif x1 - x2 > 0 and x1 - x2 > np.pi:
			xres = x1 + self.xmax
		elif x1 - x2 < 0 and x2 - x1 < np.pi:
			xres = x1 + self.xmax
		else:
			xres = x1 - self.xmax

		if np.abs(gam1 - gam2) <= self.gammax:
			gamres = gam2
		elif gam2 > gam1:
			gamres = gam1 + self.gammax
		else:
			gamres = gam1 - self.gammax
		if gamres > self.maximumClimbingAngle:
			gamres = self.maximumClimbingAngle
		if gamres < self.maximumSubductionAngle:
			gamres = self.maximumSubductionAngle

		Rq2qNext = self.distanceCost(q, qNext)
		deltax = Rq2qNext * np.cos(gamres) * np.cos(xres)
		deltay = Rq2qNext * np.cos(gamres) * np.sin(xres)
		deltaz = Rq2qNext * np.sin(gamres)

		qNext = q + np.array([deltax, deltay, deltaz])
		return x1, gam1, xres, gamres, qNext

	def checkCollision(self, q):
		for i in range(self.numberOfSphere):
			if self.distanceCost(q, self.obstacle[i, :]) <= self.Robstacle[i]:
				return np.array([0,0,i])
		for i in range(self.numberOfCylinder): 
			if 0 <= q[2] <= self.cylinderH[i] and self.distanceCost(q[0:2], self.cylinder[i, :]) <= self.cylinderR[i]:
				return np.array([0,1,i])
		return np.array([1,-1, -1])

	@staticmethod
	def distanceCost(point1, point2):  
		return np.sqrt(np.sum((point1 - point2) ** 2))

	@staticmethod
	def angleVec(vec1, vec2):  
		temp = np.dot(vec1, vec2) / np.sqrt(np.sum(vec1 ** 2)) / np.sqrt(np.sum(vec2 ** 2))
		temp = np.clip(temp,-1,1)  
		theta = np.arccos(temp)
		return theta

	@staticmethod
	def getUnitVec(vec):   
		unitVec = vec / np.sqrt(np.sum(vec ** 2))
		return unitVec

	def calculateLength(self):
		sum = 0
		for i in range(self.path.shape[0] - 1):
			sum += apf.distanceCost(self.path[i, :], self.path[i + 1, :])
		return sum

	def drawEnv(self):
		fig = plt.figure()
		self.ax=Axes3D(fig)
		plt.grid(True)  
		self.ax.scatter3D(self.qgoal[0], self.qgoal[1], self.qgoal[2], marker='o', color='red', s=100, label='Goal')
		self.ax.scatter3D(self.x0[0], self.x0[1], self.x0[2], marker='o', color='blue', s=100, label='Start')
		for i in range(self.Robstacle.shape[0]): 
			self.drawSphere(self.obstacle[i, :], self.Robstacle[i])
		for i in range(self.cylinder.shape[0]):  
			self.drawCylinder(self.cylinder[i,:],self.cylinderR[i], self.cylinderH[i])
		plt.legend(loc='best')  
		plt.grid()
		self.ax.set_xlim3d(left = 0, right = 10)
		self.ax.set_ylim3d(bottom=0, top=10)
		self.ax.set_zlim3d(bottom=0, top=10)

	def drawSphere(self, center, radius):   
		u = np.linspace(0, 2 * np.pi, 40)
		v = np.linspace(0, np.pi, 40)
		x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
		y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
		z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
		h = self.ax.plot_wireframe(x, y, z, cstride=4, color='b')
		return h

	def drawCylinder(self, center, radius, height):  
		u = np.linspace(0, 2 * np.pi, 30)  
		h = np.linspace(0, height, 20) 
		x = np.outer(center[0] + radius * np.sin(u), np.ones(len(h)))  
		y = np.outer(center[1] + radius * np.cos(u), np.ones(len(h)))  
		z = np.outer(np.ones(len(u)), h)  
		# Plot the surface
		self.ax.plot_surface(x, y, z)  

	def saveCSV(self):  
		np.savetxt('./data_csv/pathMatrix1.csv', self.path, delimiter=',')
		np.savetxt('./data_csv/obstacleMatrix.csv', self.obstacle, delimiter=',')
		np.savetxt('./data_csv/RobstacleMatrix.csv', self.Robstacle, delimiter=',')
		np.savetxt('./data_csv/cylinderMatrix.csv', self.cylinder, delimiter=',')
		np.savetxt('./data_csv/cylinderRMatrix.csv', self.cylinderR, delimiter=',')
		np.savetxt('./data_csv/cylinderHMatrix.csv', self.cylinderH, delimiter=',')

		np.savetxt('./data_csv/start.csv', self.x0, delimiter=',')
		np.savetxt('./data_csv/goal.csv', self.qgoal, delimiter=',')

	def drawPath(self):   
		self.ax.plot3D(self.path[:,0],self.path[:,1],self.path[:,2],color="deeppink",linewidth=2,label = 'AUV path')

	def loop(self):             
		q = self.x0.copy()
		qBefore = [None, None, None]
		eta1List = [0.2 for i in range(self.obstacle.shape[0])]
		eta2List = [0.2 for i in range(self.cylinder.shape[0])]
		for i in range(500):
			qNext = self.getqNext(self.epsilon0, eta1List, eta2List, q,qBefore)
			qBefore = q
			q = qNext
			if self.distanceCost(qNext,self.qgoal) < self.threshold:
				self.path = np.vstack((self.path,self.qgoal))
				break

if __name__ == "__main__":
	apf = APF()
	apf.loop()
	apf.saveCSV()
	print('distance：',apf.calculateLength())