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

def memoizeIso(f):
	cache = {}
	def ig(graph, v):
		i = (tuple(graph.V), tuple(graph.E), v)
		if i not in cache:
			cache[i] = f(graph, v)
		return cache[i]
	return ig

def memoize2(f):
	global acache	

	def a(graph, player, alpha = -1, beta = 2, maximizingPlayer=1):
		
		
		d = (tuple(graph.V), tuple(graph.E), player, alpha, beta, maximizingPlayer)
		if d not in acache:
			acache[d] = f(graph, player, alpha, beta, maximizingPlayer)
			#print (cache)
		return acache[d]
	return a

def memoize(f):	
	cache = {}
	def g2(graph, player, depth = 0, alpha = (10000, ), beta = (-10000, )):			
		d = (tuple(graph.V), tuple(graph.E), depth)
		if d not in cache:
			cache[d] = f(graph, player, depth, alpha, beta)
			#print (cache)
		return cache[d]
	return g2

def memoizeM(f):
	global mcache
	def gM(m, player, alpha = -1, beta = 2, maximizingPlayer=1):
		i = (tuple(m.keys()), tuple(m.values()), player, alpha, beta, maximizingPlayer)
		if i not in mcache:
			mcache[i] = f(m, player, alpha, beta, maximizingPlayer)
		return mcache[i]
	return gM

def merge(g1, g2, e):
	#print (g1)
	#print(g2)
	V = g1.V | g2.V
	E = g1.E | g2.E
	E.add(e)
	#print(V, E)
	return util.Graph(V,E)

def ridIso(graph):
	to_remove = {u for u in graph.V if isIsolated(graph, u)}
	graph.V.difference_update(to_remove)

def cull(graph):
	vdict = PercolationPlayer.sortVbyE(graph)
	for v, w in zip(vdict.keys(), vdict.values()):
		if w == 0 or w == 1:
			graph.V.remove(v)

		

def findMinimumNeighborhood(graph, player):
	sol = set([])
	seenV = set([])
	for v in graph.V:
		if v == player:
			v.add(v)
			seenV.add(Neighbors(graph, v))

def isCover(graph, k):
	maxn = 40
	V = len(graph.V)
	E = len(graph.E)
	Set = (1 << k) - 1
	limit = 1 << V


	EdgeList = [(e.a.index, e.b.index) for e in graph.E]
	vis = [[0] * maxn for i in range(maxn)]
	while Set < limit:
		
		vis = [[0] * maxn for i in range(maxn)]

		cnt = 0

		j = 1
		v = 0
		while j < limit:
			if Set & j:
				for k in range(0, V + 1):
					if (v, k) in EdgeList or (k, v) in EdgeList and not vis[v][k]:
						vis[v][k] = 1
						vis[k][v] = 1
						cnt += 1
			j = j << 1
			v+=1
			if (cnt == E):
				return True, vis

			c = Set & -Set
			r = Set + c
			Set = (((r ^ Set) >> 2) // c) | r
		return False, vis

def findMinCover(graph):
	n = len(graph.V)
	m = len(graph.E)
	left = 1
	right = n
	while (right > left):
		mid = (left + right) >> 1
		c = isCover(graph, mid)
		if not c[0]:
			left = mid + 1
		else:
			right = mid

	return left


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
		if (vE & 1):
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

def turn1M(matrix, player, alpha = -1, beta = 2, maximizingPlayer = 1):
	global cDict
	global glen


				
	best = -1
	vmax = None
	
	for v in matrix.keys():
		if cDict[v] == player:
			m = copy.copy(matrix)
			PercolateMatrix(m, v)
			val = auxwinnableM(m, player, alpha, beta, 0)[0]

			if val > best:
				best = val
				vmax = v

			if alpha > best:
				alpha = best

			if beta <= alpha:
				break	

	return best, vmax

@memoizeM
def auxwinnableM(matrix, player, alpha = -1, beta = 2, maximizingPlayer = 1):
	global cDict

	if isLoss(matrix, player, maximizingPlayer):
		return 1 - maximizingPlayer, None
	
	if maximizingPlayer:
					
		best = -1
		vmax = None
		
		for v in matrix.keys():
			if matrix[v] and cDict[v] == player:
				m = copy.copy(matrix)
				PercolateMatrix(m, v)
				val = auxwinnableM(m, player, alpha, beta, 0)[0]

				if val > best:
					best = val
					vmax = v

				if alpha > best:
					alpha = best

				if beta <= alpha:
					break	
		return best, vmax

	else:
		
		
		best = 2
		vmin = None

		for v in matrix.keys():
			if matrix[v] and cDict[v] != player:
				m = copy.copy(matrix)
				PercolateMatrix(m, v)
				val = auxwinnableM(m, player, alpha, beta, 1)[0]

				if val < best:
					best = val
					vmin = v

				if beta < best:
					beta = best

				if beta <= alpha:
					break

		
		return best, vmin

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
	val0 = 0
	val1 = 0
	val2 = 0
	val3 = 0


	#val1 = amt of colored vertices, -1 if me, +1 if not
	#val2 = good / bad edges, +1 is good, -1 if bad edge
	#val3 = # of edges (+1) for each edge
	c = cDict[vindex]
	binVindex = (1 << vindex)
	for v in matrix.keys():
		if v:
			e = matrix[v] 
			if cDict[v] == player:
				val1 -= 1
			else:
				val1 += 1
			if e & binVindex:
				val3 += 1
				if c == cDict[v]:
					val2 += 1
				else:
					val2 -= 1

	return (val0, val1, val2, val3)

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

	if vremoveIsLoss(matrix, player) and not depth % 2:
		return (2, 0, 0, 0), None
	elif vremoveIsLoss(matrix, 1 - player) and depth % 2:
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
			if matrix[v] and cDict[v] == player:
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
	
	def sortVbyE(graph):
		ordered = []
		vdict = {v: 0 for v in graph.V}
		for e in graph.E:
			vdict[e.a] -= 1
			vdict[e.b] -= 1
		return vdict
		

	def sortE(graph):
		vdict = PercolationPlayer.sortVbyE(graph)
		ordered = []
		count = 0
		for e in graph.E:
			heapq.heappush(ordered, (vdict[e.a] + vdict[e.b], count, e))
			count += 1
		return ordered
	
	def eDict(graph):
		vdict = PercolationPlayer.sortVbyE(graph)
		edict = {e: vdict[e.a] + vdict[e.b] for e in graph.E}
		return edict

	def Kruskalls(G, player = -1):
		g = copyGraph(G)
		ridIso(g)
		glen = len(g.V)
		F = [util.Graph([v], []) for v in g.V]
		S = PercolationPlayer.sortE(g)
		x = 0
		

		while S and len(F) > 1:
			#print(x)
			newE = heapq.heappop(S)[2]
			if not (newE.a.color == 1 - player and newE.b.color == 1 - player):
			#print (newE)
				graphV1, graphV2 = util.Graph([], []), util.Graph([], [])
				for graph in F:
					if newE.a in graph.V:
						graphV1 = graph
					elif newE.b in graph.V: 
						graphV2 = graph
				newTree = merge(graphV1, graphV2, newE)

				if len(newTree.V) == glen:
					return [newTree]
				if graphV1.V and graphV2.V:
					F.remove(graphV1)
					F.remove(graphV2)
					F.append(newTree)
				x += 1
		return F

	def VCover(graph, player):
		C = set([])
		g = copy.copy(graph)
		Eprime = PercolationPlayer.sortE(g)

		while Eprime:
			E = heapq.heappop(Eprime)[2]

			V1 = E.a
			V2 = E.b
			C.add(E.a)
			C.add(E.b)
			Eprimecopy = copy.copy(Eprime)
			for e in Eprimecopy:
				if e[2].a == V1 or e[2].a == V2 or e[2].b == V1 or e[2].b == V2:
					Eprime.remove(e)
		return C



	# `graph` is an instance of a Graph, `player` is an integer (0 or 1).
# Should return a vertex `v` from graph.V where v.color == -1
	def ChooseVertexToColor(graph, player):


		'''moves  = PercolationPlayer.VCover(graph, player)
		for v in moves:
			if v.color == -1:
				return util.GetVertex(graph, v.index)'''

		#print("no new")
		f = PercolationPlayer.Kruskalls(graph, player)
		#print(type(f))

		for t in f:
			vdict = PercolationPlayer.sortVbyE(t)
			q = []
			for v in t.V:
				if v.color == -1:
					heapq.heappush(q, (vdict[v], v.index))
			if q: 

				return util.GetVertex(graph, heapq.heappop(q)[1])

		#print("no")

		#print("did not")
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
				#if val2 == val1:
					#val2 = val2 * 2 

		

				#g = copyGraph(graph)
				#util.GetVertex(g, v.index).color = player
				#onlyColored(g)
				#size = len(graph.V)
				
				'''if size <= 10:
					w = PercolationPlayer.auxwinnable(graph, player)
				#print(w)
				#print (graph.GetVertex(w[1]))
					val0 = -1 * w[0]			
				else:
					val0 = (1000,)
					for v1 in g.V:
						if v1.color == player:				
							val0 = min(val0, PercolationPlayer.heuristic(g, v1, player))'''				
				heapq.heappush(heap, (0, 0, val2, val1, v.index))
		move = util.GetVertex(graph, heapq.heappop(heap)[4])
		#print(move)
		return move

# `graph` is an instance of a Graph, `player` is an integer (0 or 1).
# Should return a vertex `v` from graph.V where v.color == player
# Right now, this is an pretty inefficient way that finds the best move off what vertexes is connected to.
# Essentially, if a vertex is connected to a Vertex that is the same color as itself, it adds one to the value
# on the other hand, if it connected to a Vertex that is a different color, then its subtracts 1
# Then is finds the bestmove(like if I was finding the biggest from a list)
# Winrate against Random is ~60%.
# ignore the rest for now, it doesn't really do anything yet. 
	def ChooseVertexToRemove(graph, player):
		global acache
		acache.clear()
		size = len(graph.V)
		graph = GoodGraph(graph.V, graph.E)
		if size <= 0:

			w = wrapperM(graph, player)
			if w[0]:

		#print(w)
		#print (graph.GetVertex(w[1]))
				return util.GetVertex(graph, w[1])
		elif size <= 0:
			m = PercolationPlayer.vremove(graph, player)
			return util.GetVertex(graph, m[1].index)
		
		elif size <= 0:
			m = vremoveWrapper2(graph, player)
			return util.GetVertex(graph, m[1])
		move = (10000, )
		for v in graph.V:
			if v.color == player:
				
				move = min(move, PercolationPlayer.heuristic(graph,  v, player))

		return util.GetVertex(graph, move[4])
			
		
			
	@memoize
	def vremove(graph, player, depth = 0, alpha = (10000, ), beta = (-10000, )):
		#to do

		if graph.isWin1(player) == -1 and not depth % 2:
			return (2, 0, 0, 0), None
		elif graph.isWin1(1 - player) == -1 and depth % 2:
			return (-2, 0, 0, 0), None	
		elif depth == 2:
			move = (10000, )
			for v in graph.V:
				if v.color == player:
					move = min(move, PercolationPlayer.heuristic2(graph, v, player))
			return move, None
		
		if not depth % 2:
			
			
			best = (10000, )
			vmax = None
					
			for v in graph.V:
				if v.color == player:
					g = copyGraph(graph)
					g.Percolate(v)
					val = PercolationPlayer.vremove(g, player, depth + 1, alpha, beta)[0]
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
			for v in graph.V:
				if v.color != player:
					g = copyGraph(graph)
					g.Percolate(v)
					val = PercolationPlayer.vremove(g, player, depth + 1, alpha, beta)[0]
					if val > best:
						best = val
						vmax = v
					
					if beta > best:
						beta = best

					if beta >= alpha:
						break
			
			return best, vmin





	@memoize2
	def auxwinnable(graph, player, alpha = -1, beta = 2, maximizingPlayer = 1):
		if graph.isWin(player, maximizingPlayer):
			#print (1 - maximizingPlayer, graph)
			return 1 - maximizingPlayer, None
		vHeap = PercolationPlayer.orderV(graph, player, maximizingPlayer)
		#print (vHeap)
		if maximizingPlayer:
			
			
			best = -1
			vmax = None
			
			while vHeap:

				v = heapq.heappop(vHeap)[4]
				g = copyGraph(graph)
				g.Percolate(util.GetVertex(g, v))
				val = PercolationPlayer.auxwinnable(g, player, alpha, beta, 0)[0]

				if val > best:
					best = val
					vmax = v

				if alpha > best:
					alpha = best

				if beta <= alpha:
					break	

			return best, vmax

		else:

			
			best = 2
			vmin = None

			while vHeap:
				
				v = heapq.heappop(vHeap)[4]
				g = copyGraph(graph)
				g.Percolate(util.GetVertex(g, v))
				val = PercolationPlayer.auxwinnable(g, player, alpha, beta, 1)[0]

				if val < best:
					best = val
					vmin = v

				if beta < best:
					beta = best

				if beta <= alpha:
					break

			
			return best, vmin

	@memoize2
	def auxwinnable2(graph, player, alpha = -1, beta = 2, maximizingPlayer = 1):
		if PercolationPlayer.isWin(graph, player, maximizingPlayer):
			return 1 - maximizingPlayer, None
		
		if maximizingPlayer:
						
			best = -1
			vmax = None
			
			for v in graph.V:
				if v.color == player:
					g = copyGraph(graph)
					g.Percolate(util.GetVertex(g, v))
					val = PercolationPlayer.auxwinnable2(g, player, alpha, beta, 0)[0]

					if val > best:
						best = val
						vmax = v

					if alpha > best:
						alpha = best

					if beta <= alpha:
						break	

			return best, vmax

		else:
			
			
			best = 2
			vmin = None

			for v in graph.V:
				if v.color != player:
					g = copyGraph(graph)
					g.Percolate(util.GetVertex(g, v))
					val = PercolationPlayer.auxwinnable2(g, player, alpha, beta, 1)[0]

					if val < best:
						best = val
						vmin = v

					if beta < best:
						beta = best

					if beta <= alpha:
						break

			
			return best, vmin

	#@functools.lru_cache(maxsize=None)
	#@memoize
	def winnable(graph, alpha = (-1, None), beta = (2, None), v = None, maximizingPlayer = 0):
		if PercolationPlayer.isWin2(graph, maximizingPlayer):
			#print("v: " + str(v))
			return (1 - maximizingPlayer, v)

		if maximizingPlayer:

			best = (-1, None)
			
			for v in graph.V:
				if v.color == maximizingPlayer:
					g = copyGraph(graph)
					g.Percolate(v)
					val = PercolationPlayer.winnable(g, alpha, beta, v, 0)
					if val[0] > best[0]:
						best = val

					if alpha[0] > best[0]:
						alpha = best

					if beta[0] <= alpha[0]:
						break		

			return best

		else:

			best = (2, None)

			for v in graph.V:
				if v.color == maximizingPlayer:
					g = copyGraph(graph)
					g.Percolate(v)
					val = PercolationPlayer.winnable(g, alpha, beta, v, 1)

					if val[0] < best[0]:
						best = val

					if beta[0] < best[0]:
						beta = best

					if beta[0] <= alpha[0]:
						break

			
			return best
	def orderV(graph, player, maximizingPlayer):
		if maximizingPlayer:
			p = player
		else:
			p = 1 - player

		vertexHeap = [] 
		for v in graph.V:
			if v.color == p: 
				heapq.heappush(vertexHeap, PercolationPlayer.heuristic(graph, v, p))
		#print(vertexHeap)
		return vertexHeap
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

	def heuristic2(graph, v, player):


		g = copyGraph(graph)
		g.Percolate(v)
		val0 = g.isWin1(1 - player)
		if val0 == -1:
			return (val0, )
		
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

		return (val0, val1, val2, val3)

	def heuristic3(graph, v, player):

		g = copyGraph(graph)
		g.Percolate(v)

		
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
				elif va.color == 1 - player:
					val1 +=1
			if vb not in seenV:
				seenV.add(vb)
				if vb.color == player:
					val1 -=1
				elif vb.color == 1 - player:
					val1 +=1

		return (val1, val2, val3)
	
	@memoizePoint
	def point(e, c):
		if e.a.color == c and e.b.color == c:
			return 1
		else:
			return -1 


		


	


def onlyColored(graph):
	E1 = copy.copy(graph.E)
	for e in E1:
		eac = e.a.color == -1
		ebc = e.b.color == -1
		if eac or ebc:
			graph.E.remove(e)
		if eac:
			if e.a in graph.V:
				graph.V.remove(e.a)
		if ebc:
			if e.b in graph.V:
				graph.V.remove(e.b)


	


def copyGraph(graph):
	Vs = copy.copy(graph.V)
	Es = copy.copy(graph.E)

	return GoodGraph(Vs, Es)
	



# Feel free to put any personal driver code here.
def main():
	pass





if __name__ == "__main__":
    main()
