import random
import itertools
import copy
import sys
import traceback

import time
import signal
import errno
import percolator
import percolator2
from util import *
w= 0 
g = 0

# Removes the given vertex v from the graph, as well as the edges attached to it.
# Removes all isolated vertices from the graph as well.
def Percolate(graph, v):
    # Get attached edges to this vertex, remove them.
    for e in IncidentEdges(graph, v):
        graph.E.remove(e)
    # Remove this vertex.
    graph.V.remove(v)
    # Remove all isolated vertices.
    to_remove = {u for u in graph.V if len(IncidentEdges(graph, u)) == 0}
    graph.V.difference_update(to_remove)

class TimeoutError(Exception):
    pass

class Timeout:
    def __init__(self, seconds=1, error_message="Timeout of {0} seconds hit"):
        self.seconds = seconds
        self.error_message = error_message.format(seconds)
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


# This is the main game loop.
def PlayGraph(s, t, graph):
    global w
    players = [s, t]
    active_player = 0

    # Phase 1: Coloring Phase
    while any(v.color == -1 for v in graph.V):
        # First, try to just *run* the player's code to get their vertex.
        try:
            with Timeout():
                chosen_vertex = players[active_player].ChooseVertexToColor(copy.deepcopy(graph), active_player)

        # If user code does not return within appropriate timeout, select random action.
        except TimeoutError as e:
            print(e)
            traceback.print_exc(file=sys.stdout)
            chosen_vertex = RandomPlayer.ChooseVertexToColor(copy.deepcopy(graph), active_player)
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            return 1 - active_player
        # Next, check that their output was reasonable.
        try:
            original_vertex = GetVertex(graph, chosen_vertex.index)
            if not original_vertex:
                return 1 - active_player
            if original_vertex.color != -1:
                return 1 - active_player
            # If output is reasonable, color this vertex.
            original_vertex.color = active_player
        # Only case when this should fire is if chosen_vertex.index does not exist or similar error.
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            return 1 - active_player

        # Swap current player.
        active_player = 1 - active_player

    # Check that all vertices are colored now.
    assert all(v.color != -1 for v in graph.V)

    #w += percolator.wrapperM(percolator.GoodGraph(graph.V,graph.E), 0)[0]
    # Phase 2: Removal phase
    # Continue while both players have vertices left to remove.
    while len([v for v in graph.V if v.color == active_player]) > 0:
        # First, try to just *run* the removal code.
        try:
            with Timeout():
                chosen_vertex = players[active_player].ChooseVertexToRemove(copy.deepcopy(graph), active_player)

        # If user code does not return within appropriate timeout, select random action.
        except TimeoutError as e:
            print(e)
            traceback.print_exc(file=sys.stdout)
            chosen_vertex = RandomPlayer.ChooseVertexToRemove(copy.deepcopy(graph), active_player)
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            return 1 - active_player
        # Next, check that their output was reasonable.
        try:
            original_vertex = GetVertex(graph, chosen_vertex.index)
            if not original_vertex:
                return 1 - active_player
            if original_vertex.color != active_player:
                return 1 - active_player
            # If output is reasonable, remove ("percolate") this vertex + edges attached to it, as well as isolated vertices.
            Percolate(graph, original_vertex)
        # Only case when this should fire is if chosen_vertex.index does not exist or similar error.
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            return 1 - active_player
        # Swap current player
        active_player = 1 - active_player

    # Winner is the non-active player.
    return 1 - active_player


# This method generates a binomial random graph with 2k vertices
# having probability p of an edge between each pair of vertices.
def BinomialRandomGraph(k, p):
    v = {Vertex(i) for i in range(2 * k)}
    e = {Edge(a, b) for (a, b) in itertools.combinations(v, 2) if random.random() < p}
    return Graph(v, e)


# This method creates and plays a number of random graphs using both passed in players.
def PlayBenchmark(p1, p2, iters):
    global g
    graphs = (
        BinomialRandomGraph(random.randint(2,20), random.random())
        for _ in range(iters)
    )
    wins = [0, 0]
    for graph in graphs:
        
        if len(graph.E) == 1 or len(graph.E) == 0:
            g+=1
        g1 = copy.deepcopy(graph)
        g2 = copy.deepcopy(graph)
        # Each player gets a chance to go first on each graph.
        winner_a = PlayGraph(p1, p2, g1)
        #if winner_a == 1:
            #print(g2)
        wins[winner_a] += 1
        winner_b = PlayGraph(p2, p1, g2)
        wins[1-winner_b] += 1
    return wins

def part1(graph, s, t):
    global w
    players = [s, t]
    active_player = 0

    # Phase 1: Coloring Phase
    while any(v.color == -1 for v in graph.V):
        # First, try to just *run* the player's code to get their vertex.
        try:
            with Timeout():
                chosen_vertex = players[active_player].ChooseVertexToColor(copy.deepcopy(graph), active_player)

        # If user code does not return within appropriate timeout, select random action.
        except TimeoutError as e:
            print(e)
            traceback.print_exc(file=sys.stdout)
            chosen_vertex = RandomPlayer.ChooseVertexToColor(copy.deepcopy(graph), active_player)
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            return 1 - active_player
        # Next, check that their output was reasonable.
        try:
            original_vertex = GetVertex(graph, chosen_vertex.index)
            if not original_vertex:
                return 1 - active_player
            if original_vertex.color != -1:
                return 1 - active_player
            # If output is reasonable, color this vertex.
            original_vertex.color = active_player
        # Only case when this should fire is if chosen_vertex.index does not exist or similar error.
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            return 1 - active_player

        # Swap current player.
        active_player = 1 - active_player

# This is a player that plays a legal move at random.
class RandomPlayer:
    # These are "static methdods" - note there's no "self" parameter here.
    # These methods are defined on the blueprint/class definition rather than
    # any particular instance.
    def ChooseVertexToColor(graph, active_player):
        return random.choice([v for v in graph.V if v.color == -1])

    def ChooseVertexToRemove(graph, active_player):
        return random.choice([v for v in graph.V if v.color == active_player])


if __name__ == "__main__":
    # NOTE: we are not creating INSTANCES of these classes, we're defining the players
    # as the class itself. This lets us call the static methods.
    p1 = percolator.PercolationPlayer
    # Comment the above line and uncomment the next two if
    # you'd like to test the PercolationPlayer code in this repo.
    # from percolator import PercolationPlayer
    # p1 = PercolationPlayer
    p2 = RandomPlayer
    iters = 200
    wins = [1,1]
    wins = PlayBenchmark(p1, p2, iters)
    print(wins)
    print(
        "Player 1: {0} Player 2: {1}".format(
            1.0 * wins[0] / sum(wins), 1.0 * wins[1] / sum(wins)
        )
    )
    tw = (wins[0] - g) / (wins[1] + wins[0] - (2 * g))
    print('true wr: ' + str(tw))
    print(w)
    print(g)

    l1 = []
    l2 = []
    for i in range(10):
        break
        g = BinomialRandomGraph(9, 0.5)
        part1(g, RandomPlayer, RandomPlayer)
        #print(g)
        g = percolator.GoodGraph(g.V, g.E)
        g.Percolate(GetVertex(g, 1))
        #print(g)
        percolator.PercolationPlayer.vremove(g, 0)
        l1.append((percolator.PercolationPlayer.vremove(g, 0)[0], percolator.PercolationPlayer.vremove(g, 0)[1].index))
        l2.append(percolator.vremoveWrapper2(g, 0))
    #print(l1)
    #print(l2)
    #print(l1 == l2)


   





