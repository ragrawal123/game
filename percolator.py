import heapq
import util
import random
import copy
import functools
import heapq
import sys

import time
cDict = None
glen = None
acache = {}
mcache = {}
class GoodGraph:
	def __init__(self, v, e):
		self.V = set(v)
		self.E = set(e)

	def __repr__(self):
		return "Graph({0}, {1})".format(self.V, self.E)

	def __deepcopy__(self, memo):
		V = {v.index: Vertex(v.index, v.color) for v in self.V}
		E = [Edge(V[e.a.index], V[e.b.index]) for e in self.E]
		return Graph(V.values(), E)

	def isIsolated(self, v):
		for e in self.E:
			if e.a == v or e.b == v:
				return False
		return True

	def Percolate(self, v):
		# Get attached edges to this vertex, remove them.
		

		E1 = copy.copy(self.E)
		for e in E1:
			v1 = e.a
			v2 = e.b
			if v1 == v:
				self.E.remove(e)
			elif v2 == v:
				self.E.remove(e)
		
		self.V.remove(v)

		V1 = copy.copy(self.V)
		for v1 in V1:
			if self.isIsolated(v1):
				self.V.remove(v1)
        # Remove this vertex.




	def isWin(self, player, maximizingPlayer):

		if maximizingPlayer:
			p = player
		else:
			p = 1 - player

		return self.isWin2(p) 


		

	def isWin1(self, player):
		for v in self.V:
			if v.color == player:
				return 0
		return -1 

	def isWin2(self, player):

		for v in self.V:
			if v.color == player:
				return False
		return True

def isIsolated(graph, v):
		for e in graph.E:
			if e.a == v or e.b == v:
				return False
		return True

def memoizePoint(f):
	cache = {}
	def pg(e, c):
		i = (e, c)
		if i not in cache:
			cache[i] = f(e, c)
		return cache[i]
	return pg

def memoizeM(f):
	global mcache
	def gM(m, player, alpha = (-1, ), beta = (2, ), maximizingPlayer=1):
		i = (tuple(m.keys()), tuple(m.values()), player, alpha, beta, maximizingPlayer)
		if i not in mcache:
			mcache[i] = f(m, player, alpha, beta, maximizingPlayer)
		return mcache[i]
	return gM


def createAMatrix(graph):
	l = {v.index: 0 for v in graph.V}
	#print(graph)
	#print(l)
	for e in graph.E:
		v1 = e.a.index
		v2 = e.b.index
		l[v1] = (1 << v2) | l[v1]
		l[v2] = (1 << v1) | l[v2]

	return l

def PercolateMatrix(matrix, vindex):
	t = ~ ( 1 << vindex)
	vE = matrix[vindex]
	i = 0
	while vE:	
		#print(vE)
		if vE & 1:
			#print(matrix)
			#print(i)
			matrix[i] = t & matrix[i]
		vE = vE >> 1
		i += 1
	matrix[vindex] = 0

def colorDict(graph):
	return {v.index: v.color for v in graph.V}


def isLoss(matrix, player, maximizingPlayer):
	global cDict
	global glen
	if maximizingPlayer:
		p = player
	else:
		p = 1 - player

	for i in matrix.keys():
		if matrix[i] and cDict[i] == p:
			return False
	return True



def wrapperM(graph, player):
	global cDict
	global glen
	global mcache
	mcache.clear()
	cDict = colorDict(graph)
	#print(cDict)
	glen = len(graph.V)
	m = createAMatrix(graph)
	#print(m)
	#PercolateMatrix(m, 0)
	#print(m)
	#print(isLoss(m, 0, 0))
	return turn1M(m, player)

def turn1M(matrix, player, alpha = (-1, ), beta = (2, ), maximizingPlayer = 1):
	global cDict
	global glen

	a = 0
	s = 0
				
	best = (-1, )
	vmax = None
	
	for v in matrix.keys():
		if cDict[v] == player:
			m = copy.copy(matrix)
			PercolateMatrix(m, v)
			val = auxwinnableM(m, player, alpha, beta, 0)[0]

			s += val[0]
			a += 1

			if val > best:
				best = val
				vmax = v

			if alpha > best:
				alpha = best

			if beta <= alpha:
				break	

	return (best[0], s / a), vmax

@memoizeM
def auxwinnableM(matrix, player, alpha = (-1, ), beta = (2, ), maximizingPlayer = 1):
	global cDict
	a = 0
	s = 0
	if isLoss(matrix, player, maximizingPlayer):
		return (1 - maximizingPlayer, 1 - maximizingPlayer), None
	
	if maximizingPlayer:
					
		best = (-1, )
		vmax = None
		
		for v in matrix.keys():
			if matrix[v] and cDict[v] == player:
				m = copy.copy(matrix)
				PercolateMatrix(m, v)
				val = auxwinnableM(m, player, alpha, beta, 0)[0]
				


				s += val[0]
				a += 1

				if val > best:
					best = val
					vmax = v

				if alpha > best:
					alpha = best

				if beta <= alpha:
					break	
		return (best[0], s / a), vmax

	else:
		
		
		best = (2, )
		vmin = None

		for v in matrix.keys():
			if matrix[v] and cDict[v] != player:
				m = copy.copy(matrix)
				PercolateMatrix(m, v)
				val = auxwinnableM(m, player, alpha, beta, 1)[0]


				if val < best:
					best = val
					vmin = v

				s += val[0]
				a += 1

				if beta < best:
					beta = best

				if beta <= alpha:
					break

		return (best[0], s / a), vmin

def vremoveIsLoss(matrix, player):
	for i in matrix.keys():
		if matrix[i] and cDict[i] == player:
			return False
	return True

def heuristicM(matrix, vindex, player):
	global cDict
	m = copy.copy(matrix)
	PercolateMatrix(m, vindex)
	if vremoveIsLoss(m, 1 - player):
		return (-1, )

	val1 = 0
	val2 = 0
	val3 = 0


	#val1 = amt of colored vertices, -1 if me, +1 if not
	#val2 = good / bad edges, +1 is good, -1 if bad edge
	#val3 = # of edges (+1) for each edge
	c = cDict[vindex]
	binVindex = (1 << vindex)
	for v in matrix.keys():
		if matrix[v]:
			if cDict[v] == player:
				val1 -= 1
			else:
				val1 += 1
			if  matrix[v] & binVindex:
				val3 += 1
				if c == cDict[v]:
					val2 += 1
				else:
					val2 -= 1

	return (0, val1, val2, val3)

def vremoveWrapper(graph, player):
	global cDict

	cDict = colorDict(graph)
	#print(cDict)
	m = createAMatrix(graph)
	return vremoveMT1(m, player)
	#return heuristicM(m, 0, player)

def vremoveWrapper2(graph, player):
	global cDict

	cDict = colorDict(graph)
	#print(cDict)
	m = createAMatrix(graph)
	return vremoveMT1(m, player, 2)
	#return heuristicM(m, 0, player)

def vremoveMT1(matrix, player, depth = 0, alpha = (10000, ), beta = (-10000, )):
	best = (10000, )
	vmax = None
			
	for v in matrix.keys():
		if cDict[v] == player:
			m = copy.copy(matrix)
			PercolateMatrix(m, v)
			val = vremoveM(m, player, depth + 1, alpha, beta)[0]
			if val < best:
				best = val
				vmax = v

			if alpha < best:
				alpha = best

			if beta >= alpha:
				break
	return best, vmax

def vremoveM(matrix, player, depth = 0, alpha = (10000, ), beta = (-10000, )):
	#to do
	global cDict

	if not depth % 2 and vremoveIsLoss(matrix, player):
		return (2, 0, 0, 0), None
	elif depth % 2 and vremoveIsLoss(matrix, 1 - player):
		return (-2, 0, 0, 0), None	
	elif depth == 4:
		move = (10000, )
		for v in matrix.keys():
			if matrix[v] and cDict[v] == player:
				move = min(move, heuristicM(matrix, v, player))
		return move, None
	
	if not depth % 2:
		
		
		best = (10000, )
		vmax = None
				
		for v in matrix.keys():
			if matrix[v] and cDict[v] == player:
				m = copy.copy(matrix)
				PercolateMatrix(m, v)
				val = vremoveM(m, player, depth + 1, alpha, beta)[0]
				if val < best:
					best = val
					vmax = v

				if alpha < best:
					alpha = best

				if beta >= alpha:
					break
		return best, vmax
	
	else:

		best = (-10000, )
		vmin = None 
		for v in matrix.keys():
			if matrix[v] and cDict[v] != player:
				m = copy.copy(matrix)
				PercolateMatrix(m, v)
				val = vremoveM(m, player, depth + 1, alpha, beta)[0]
				if val > best:
					best = val
					vmax = v
				
				if beta > best:
					beta = best

				if beta >= alpha:
					break
		
		return best, vmin

class PercolationPlayer:
	

	# `graph` is an instance of a Graph, `player` is an integer (0 or 1).
# Should return a vertex `v` from graph.V where v.color == -1
	def ChooseVertexToColor(graph, player): 

		heap = []
		for v in graph.V:

			val1 = 0
			val2 = 0
			val3 = 0
			uncolored = 0 
			

			if v.color == -1:
				for e in graph.E:
					ea = e.a
					eb = e.b
					if ea == v or eb == v:
						if ea.color == player or eb.color == player:
							val1 += 2
						elif ea.color == 1 - player or eb.color == 1 - player:
							val1 += -1
						val2 -=1
			
				heapq.heappush(heap, (0, 0, val2, val1, v.index))
		move = util.GetVertex(graph, heapq.heappop(heap)[4])
		#print(move)
		return move

	def ChooseVertexToRemove(graph, player):
		size = len(graph.V)
		graph = GoodGraph(graph.V, graph.E)

		if size <= 12:

			w = wrapperM(graph, player)
			return util.GetVertex(graph, w[1])

		elif size <= 14:
			m = vremoveWrapper(graph, player)
			return util.GetVertex(graph, m[1])
		
		else:
			m = vremoveWrapper2(graph, player)
			return util.GetVertex(graph, m[1])
		

		move = (10000, )
		for v in graph.V:
			if v.color == player:
				
				move = min(move, PercolationPlayer.heuristic(graph,  v, player))

		return util.GetVertex(graph, move[4])
			
		
	def heuristic(graph, v, player):

		g = copyGraph(graph)
		g.Percolate(v)
		val0 = g.isWin1(1 - player)
		if val0 == -1:
			return (val0, 0, 0, 0, v.index)
		
		val1 = 0
		val2 = 0
		val3 = 0

		
		seenV = set([])

		c = v.color
		for e in graph.E:
			va = e.a
			vb = e.b
			if va == v or vb == v:
				val3 += 1
				val2 += PercolationPlayer.point(e, c)
			if va not in seenV:
				seenV.add(va)
				if va.color == player:
					val1 -=1
				else:
					val1 +=1
			if vb not in seenV:
				seenV.add(vb)
				if vb.color == player:
					val1 -=1
				else:
					val1 +=1

		return (val0, val1, val2, val3, v.index)

	@memoizePoint
	def point(e, c):
		if e.a.color == c and e.b.color == c:
			return 1
		else:
			return -1


def copyGraph(graph):
	Vs = copy.copy(graph.V)
	Es = copy.copy(graph.E)

	return GoodGraph(Vs, Es)

 
# Feel free to put any personal driver code here.
def main():
	pass





if __name__ == "__main__":
    main()
