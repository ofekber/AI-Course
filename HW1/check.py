import ex1_old as ex1
import search
import time
from problems import non_comp_problems
from problems import comp_problems
#from problems import sorted_t_problems


def timeout_exec(func, args=(), kwargs={}, timeout_duration=10, default=None):
    """This function will spawn a thread and run the given function
    using the args, kwargs and return the given default value if the
    timeout_duration is exceeded.
    """
    import threading

    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = default

        def run(self):
            self.result = func(*args, **kwargs)

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.is_alive():
        return default
    else:
        return it.result


def check_problem(p, search_method, timeout):
    """ Constructs a problem using ex1.create_wumpus_problem,
    and solves it using the given search_method with the given timeout.
    Returns a tuple of (solution length, solution time, solution)"""

    """ (-2, -2, None) means there was a timeout
    (-3, -3, ERR) means there was some error ERR during search """

    t1 = time.time()
    s = timeout_exec(search_method, args=[p], timeout_duration=timeout)
    t2 = time.time()

    if isinstance(s, search.Node):
        solve = s
        solution = list(map(lambda n: n.action, solve.path()))[1:]
        return len(solution), t2 - t1, solution
    elif s is None:
        return -2, -2, None
    else:
        return s


def solve_problems(problems, COUNT):
    solved = 0

    for problem in problems:
        COUNT += 1
        try:
            p = ex1.create_harrypotter_problem(problem)
        except Exception as e:
            print("Error creating problem: ", e)
            return None
        timeout = 60
        result = check_problem(
            p, (lambda p: search.astar_search(p, p.h)), timeout)
        print(COUNT, ".", "A* ", result)
        if result[2] is not None:
            if result[0] != -3:
                solved += 1

    return COUNT


def main():
    start_time = time.time()  # Start the timer
    print(ex1.ids)
    """Here goes the input you want to check"""
    COUNT = 0

    COUNT = solve_problems(non_comp_problems, COUNT)
    COUNT = solve_problems(comp_problems, COUNT)
    #solve_problems(sorted_t_problems, COUNT)

    end_time = time.time()  # End the timer
    print(f"Total runtime: {end_time - start_time:.2f} seconds")  # Print total runtime


if __name__ == '__main__':
    main()
