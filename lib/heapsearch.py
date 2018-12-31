from heapq import heappush, heappop


def solve_heapsearch(problem, max_steps=float('inf'), reset=True):
    """
    Attempts to find a given anonymous walk in graph by searching over nodes with highest q-value first
    :type problem: lib.problem.GAWProblem
    :param max_steps: maximum number of edges considered. Defines algorithm's time budget.
    :param reset: if True, resets the problem before starting search
    """
    if reset:
        problem.reset()

    heap = []  # min-heap, priority = -1 * progress
    heappush(heap, (0, problem.get_state()))
    transitions = []  # s, a, r, is_done
    best_solution = problem.path

    while len(heap):
        _, state_t = heappop(heap)
        problem.load_state(state_t)

        for vertex in problem.get_valid_actions():
            _, reward, is_done, _ = problem.step(chosen_next_vertex=vertex)
            next_state = problem.get_state()

            transitions.append((state_t, vertex, reward, is_done))
            if not is_done:
                heappush(heap, (-len(problem.path), next_state))
            elif len(problem.path) > len(best_solution):
                best_solution = problem.path
                if len(best_solution) == len(problem.walk):
                    return best_solution, transitions
            else:
                pass

            problem.load_state(state_t)

            if len(transitions) >= max_steps:
                return best_solution, transitions

    return best_solution, transitions
