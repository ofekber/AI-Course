"""Provides some utilities widely used by other modules"""

import bisect
import collections
import collections.abc
import operator
import os.path
import random
import math
import functools
from itertools import chain, combinations


# ______________________________________________________________________________
# Functions on Sequences and Iterables


def sequence(iterable):
    """Coerce iterable to sequence, if it is not already one."""
    return (iterable if isinstance(iterable, collections.abc.Sequence)
            else tuple(iterable))

def remove_all(item, seq):
    """Return a copy of seq (or string) with all occurrences of item removed."""
    if isinstance(seq, str):
        return seq.replace(item, '')
    elif isinstance(seq, set):
        rest = seq.copy()
        rest.remove(item)
        return rest
    else:
        return [x for x in seq if x != item]


def unique(seq):  # TODO: replace with set
    """Remove duplicate elements from seq. Assumes hashable elements."""
    return list(set(seq))


def count(seq):
    """Count the number of items in sequence that are interpreted as true."""
    return sum(bool(x) for x in seq)


def product(numbers):
    """Return the product of the numbers, e.g. product([2, 3, 10]) == 60"""
    result = 1
    for x in numbers:
        result *= x
    return result


def first(iterable, default=None):
    """Return the first element of an iterable or the next element of a generator; or default."""
    try:
        return iterable[0]
    except IndexError:
        return default
    except TypeError:
        return next(iterable, default)


def is_in(elt, seq):
    """Similar to (elt in seq), but compares with 'is', not '=='."""
    return any(x is elt for x in seq)


def mode(data):
    """Return the most common data item. If there are ties, return any one of them."""
    [(item, count)] = collections.Counter(data).most_common(1)
    return item


def powerset(iterable):
    """powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))[1:]


# ______________________________________________________________________________
# argmin and argmax


identity = lambda x: x

argmin = min
argmax = max


def argmin_random_tie(seq, key=identity):
    """Return a minimum element of seq; break ties at random."""
    return argmin(shuffled(seq), key=key)


def argmax_random_tie(seq, key=identity):
    """Return an element with highest fn(seq[i]) score; break ties at random."""
    return argmax(shuffled(seq), key=key)


def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items


# ______________________________________________________________________________
# Statistical and mathematical functions


def histogram(values, mode=0, bin_function=None):
    """Return a list of (value, count) pairs, summarizing the input values.
    Sorted by increasing value, or if mode=1, by decreasing count.
    If bin_function is given, map it over values first."""
    if bin_function:
        values = map(bin_function, values)

    bins = {}
    for val in values:
        bins[val] = bins.get(val, 0) + 1

    if mode:
        return sorted(list(bins.items()), key=lambda x: (x[1], x[0]),
                      reverse=True)
    else:
        return sorted(bins.items())


def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors X and Y."""
    return sum(x * y for x, y in zip(X, Y))


def element_wise_product(X, Y):
    """Return vector as an element-wise product of vectors X and Y"""
    assert len(X) == len(Y)
    return [x * y for x, y in zip(X, Y)]


def matrix_multiplication(X_M, *Y_M):
    """Return a matrix as a matrix-multiplication of X_M and arbitary number of matrices *Y_M"""

    def _mat_mult(X_M, Y_M):
        """Return a matrix as a matrix-multiplication of two matrices X_M and Y_M
        >>> matrix_multiplication([[1, 2, 3],
                                   [2, 3, 4]],
                                   [[3, 4],
                                    [1, 2],
                                    [1, 0]])
        [[8, 8],[13, 14]]
        """
        assert len(X_M[0]) == len(Y_M)

        result = [[0 for i in range(len(Y_M[0]))] for j in range(len(X_M))]
        for i in range(len(X_M)):
            for j in range(len(Y_M[0])):
                for k in range(len(Y_M)):
                    result[i][j] += X_M[i][k] * Y_M[k][j]
        return result

    result = X_M
    for Y in Y_M:
        result = _mat_mult(result, Y)

    return result


def vector_to_diagonal(v):
    """Converts a vector to a diagonal matrix with vector elements
    as the diagonal elements of the matrix"""
    diag_matrix = [[0 for i in range(len(v))] for j in range(len(v))]
    for i in range(len(v)):
        diag_matrix[i][i] = v[i]

    return diag_matrix


def vector_add(a, b):
    """Component-wise addition of two vectors."""
    return tuple(map(operator.add, a, b))


def scalar_vector_product(X, Y):
    """Return vector as a product of a scalar and a vector"""
    return [X * y for y in Y]


def scalar_matrix_product(X, Y):
    """Return matrix as a product of a scalar and a matrix"""
    return [scalar_vector_product(X, y) for y in Y]


def inverse_matrix(X):
    """Inverse a given square matrix of size 2x2"""
    assert len(X) == 2
    assert len(X[0]) == 2
    det = X[0][0] * X[1][1] - X[0][1] * X[1][0]
    assert det != 0
    inv_mat = scalar_matrix_product(1.0/det, [[X[1][1], -X[0][1]], [-X[1][0], X[0][0]]])

    return inv_mat


def probability(p):
    """Return true with probability p."""
    return p > random.uniform(0.0, 1.0)


def weighted_sample_with_replacement(n, seq, weights):
    """Pick n samples from seq at random, with replacement, with the
    probability of each element in proportion to its corresponding
    weight."""
    sample = weighted_sampler(seq, weights)

    return [sample() for _ in range(n)]


def weighted_sampler(seq, weights):
    """Return a random-sample function that picks from seq weighted by weights."""
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)

    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]


def rounder(numbers, d=4):
    """Round a single number, or sequence of numbers, to d decimal places."""
    if isinstance(numbers, (int, float)):
        return round(numbers, d)
    else:
        constructor = type(numbers)     # Can be list, set, tuple, etc.
        return constructor(rounder(n, d) for n in numbers)


def num_or_str(x):
    """The argument is a string; convert to a number if
       possible, or strip it."""
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return str(x).strip()


def normalize(dist):
    """Multiply each number by a constant such that the sum is 1.0"""
    if isinstance(dist, dict):
        total = sum(dist.values())
        for key in dist:
            dist[key] = dist[key] / total
            assert 0 <= dist[key] <= 1, "Probabilities must be between 0 and 1."
        return dist
    total = sum(dist)
    return [(n / total) for n in dist]


def norm(X, n=2):
    """Return the n-norm of vector X"""
    return sum([x**n for x in X])**(1/n)


def clip(x, lowest, highest):
    """Return x clipped to the range [lowest..highest]."""
    return max(lowest, min(x, highest))


def sigmoid_derivative(value):
    return value * (1 - value)


def sigmoid(x):
    """Return activation value of x with sigmoid function"""
    return 1/(1 + math.exp(-x))


def step(x):
    """Return activation value of x with sign function"""
    return 1 if x >= 0 else 0


def gaussian(mean, st_dev, x):
    """Given the mean and standard deviation of a distribution, it returns the probability of x."""
    return 1/(math.sqrt(2*math.pi)*st_dev)*math.e**(-0.5*(float(x-mean)/st_dev)**2)


try:  # math.isclose was added in Python 3.5; but we might be in 3.4
    from math import isclose
except ImportError:
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        """Return true if numbers a and b are close to each other."""
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def weighted_choice(choices):
    """A weighted version of random.choice"""
    # NOTE: Shoule be replaced by random.choices if we port to Python 3.6

    total = sum(w for _, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w >= r:
            return c, w
        upto += w


# ______________________________________________________________________________
# Grid Functions


orientations = EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]
turns = LEFT, RIGHT = (+1, -1)


def turn_heading(heading, inc, headings=orientations):
    return headings[(headings.index(heading) + inc) % len(headings)]


def turn_right(heading):
    return turn_heading(heading, RIGHT)


def turn_left(heading):
    return turn_heading(heading, LEFT)


def distance(a, b):
    """The distance between two (x, y) points."""
    xA, yA = a
    xB, yB = b
    return math.hypot((xA - xB), (yA - yB))


def distance_squared(a, b):
    """The square of the distance between two (x, y) points."""
    xA, yA = a
    xB, yB = b
    return (xA - xB)**2 + (yA - yB)**2


def vector_clip(vector, lowest, highest):
    """Return vector, except if any element is less than the corresponding
    value of lowest or more than the corresponding value of highest, clip to
    those values."""
    return type(vector)(map(clip, vector, lowest, highest))


# ______________________________________________________________________________
# Misc Functions


def memoize(fn, slot=None, maxsize=32):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, use lru_cache for caching the values."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args):
            return fn(*args)

    return memoized_fn


def name(obj):
    """Try to find some reasonable name for the object."""
    return (getattr(obj, 'name', 0) or getattr(obj, '__name__', 0) or
            getattr(getattr(obj, '__class__', 0), '__name__', 0) or
            str(obj))


def isnumber(x):
    """Is x a number?"""
    return hasattr(x, '__int__')


def issequence(x):
    """Is x a sequence?"""
    return isinstance(x, collections.abc.Sequence)


def print_table(table, header=None, sep='   ', numfmt='{}'):
    """Print a list of lists as a table, so that columns line up nicely.
    header, if specified, will be printed as the first row.
    numfmt is the format for all numbers; you might want e.g. '{:.2f}'.
    (If you want different formats in different columns,
    don't use print_table.) sep is the separator between columns."""
    justs = ['rjust' if isnumber(x) else 'ljust' for x in table[0]]

    if header:
        table.insert(0, header)

    table = [[numfmt.format(x) if isnumber(x) else x for x in row]
             for row in table]

    sizes = list(
            map(lambda seq: max(map(len, seq)),
                list(zip(*[map(str, row) for row in table]))))

    for row in table:
        print(sep.join(getattr(
            str(x), j)(size) for (j, size, x) in zip(justs, sizes, row)))


def open_data(name, mode='r'):
    aima_root = os.path.dirname(__file__)
    aima_file = os.path.join(aima_root, *['aima-data', name])

    return open(aima_file)


# ______________________________________________________________________________
# Expressions

# See https://docs.python.org/3/reference/expressions.html#operator-precedence
# See https://docs.python.org/3/reference/datamodel.html#special-method-names

class Expr(object):
    """A mathematical expression with an operator and 0 or more arguments.
    op is a str like '+' or 'sin'; args are Expressions.
    Expr('x') or Symbol('x') creates a symbol (a nullary Expr).
    Expr('-', x) creates a unary; Expr('+', x, 1) creates a binary."""

    def __init__(self, op, *args):
        self.op = str(op)
        self.args = args

    # Operator overloads
    def __neg__(self):
        return Expr('-', self)

    def __pos__(self):
        return Expr('+', self)

    def __invert__(self):
        return Expr('~', self)

    def __add__(self, rhs):
        return Expr('+', self, rhs)

    def __sub__(self, rhs):
        return Expr('-', self, rhs)

    def __mul__(self, rhs):
        return Expr('*', self, rhs)

    def __pow__(self, rhs):
        return Expr('**', self, rhs)

    def __mod__(self, rhs):
        return Expr('%', self, rhs)

    def __and__(self, rhs):
        return Expr('&', self, rhs)

    def __xor__(self, rhs):
        return Expr('^', self, rhs)

    def __rshift__(self, rhs):
        return Expr('>>', self, rhs)

    def __lshift__(self, rhs):
        return Expr('<<', self, rhs)

    def __truediv__(self, rhs):
        return Expr('/', self, rhs)

    def __floordiv__(self, rhs):
        return Expr('//', self, rhs)

    def __matmul__(self, rhs):
        return Expr('@', self, rhs)

    def __or__(self, rhs):
        """Allow both P | Q, and P |'==>'| Q."""
        if isinstance(rhs, Expression):
            return Expr('|', self, rhs)
        else:
            return PartialExpr(rhs, self)

    # Reverse operator overloads
    def __radd__(self, lhs):
        return Expr('+', lhs, self)

    def __rsub__(self, lhs):
        return Expr('-', lhs, self)

    def __rmul__(self, lhs):
        return Expr('*', lhs, self)

    def __rdiv__(self, lhs):
        return Expr('/', lhs, self)

    def __rpow__(self, lhs):
        return Expr('**', lhs, self)

    def __rmod__(self, lhs):
        return Expr('%', lhs, self)

    def __rand__(self, lhs):
        return Expr('&', lhs, self)

    def __rxor__(self, lhs):
        return Expr('^', lhs, self)

    def __ror__(self, lhs):
        return Expr('|', lhs, self)

    def __rrshift__(self, lhs):
        return Expr('>>', lhs, self)

    def __rlshift__(self, lhs):
        return Expr('<<', lhs, self)

    def __rtruediv__(self, lhs):
        return Expr('/', lhs, self)

    def __rfloordiv__(self, lhs):
        return Expr('//', lhs, self)

    def __rmatmul__(self, lhs):
        return Expr('@', lhs, self)

    def __call__(self, *args):
        "Call: if 'f' is a Symbol, then f(0) == Expr('f', 0)."
        if self.args:
            raise ValueError('can only do a call for a Symbol, not an Expr')
        else:
            return Expr(self.op, *args)

    # Equality and repr
    def __eq__(self, other):
        "'x == y' evaluates to True or False; does not build an Expr."
        return (isinstance(other, Expr)
                and self.op == other.op
                and self.args == other.args)

    def __hash__(self): return hash(self.op) ^ hash(self.args)

    def __repr__(self):
        op = self.op
        args = [str(arg) for arg in self.args]
        if op.isidentifier():       # f(x) or f(x, y)
            return '{}({})'.format(op, ', '.join(args)) if args else op
        elif len(args) == 1:        # -x or -(x + 1)
            return op + args[0]
        else:                       # (x - y)
            opp = (' ' + op + ' ')
            return '(' + opp.join(args) + ')'

# An 'Expression' is either an Expr or a Number.
# Symbol is not an explicit type; it is any Expr with 0 args.


Number = (int, float, complex)
Expression = (Expr, Number)


def Symbol(name):
    """A Symbol is just an Expr with no args."""
    return Expr(name)


def symbols(names):
    """Return a tuple of Symbols; names is a comma/whitespace delimited str."""
    return tuple(Symbol(name) for name in names.replace(',', ' ').split())


def subexpressions(x):
    """Yield the subexpressions of an Expression (including x itself)."""
    yield x
    if isinstance(x, Expr):
        for arg in x.args:
            yield from subexpressions(arg)


def arity(expression):
    """The number of sub-expressions in this expression."""
    if isinstance(expression, Expr):
        return len(expression.args)
    else:  # expression is a number
        return 0

# For operators that are not defined in Python, we allow new InfixOps:


class PartialExpr:
    """Given 'P |'==>'| Q, first form PartialExpr('==>', P), then combine with Q."""
    def __init__(self, op, lhs):
        self.op, self.lhs = op, lhs

    def __or__(self, rhs):
        return Expr(self.op, self.lhs, rhs)

    def __repr__(self):
        return "PartialExpr('{}', {})".format(self.op, self.lhs)


def expr(x):
    """Shortcut to create an Expression. x is a str in which:
    - identifiers are automatically defined as Symbols.
    - ==> is treated as an infix |'==>'|, as are <== and <=>.
    If x is already an Expression, it is returned unchanged. Example:
    >>> expr('P & Q ==> Q')
    ((P & Q) ==> Q)
    """
    if isinstance(x, str):
        return eval(expr_handle_infix_ops(x), defaultkeydict(Symbol))
    else:
        return x


infix_ops = '==> <== <=>'.split()


def expr_handle_infix_ops(x):
    """Given a str, return a new str with ==> replaced by |'==>'|, etc.
    >>> expr_handle_infix_ops('P ==> Q')
    "P |'==>'| Q"
    """
    for op in infix_ops:
        x = x.replace(op, '|' + repr(op) + '|')
    return x


class defaultkeydict(collections.defaultdict):
    """Like defaultdict, but the default_factory is a function of the key.
    >>> d = defaultkeydict(len); d['four']
    4
    """
    def __missing__(self, key):
        self[key] = result = self.default_factory(key)
        return result


class hashabledict(dict):
    """Allows hashing by representing a dictionary as tuple of key:value pairs
       May cause problems as the hash value may change during runtime
    """
    def __tuplify__(self):
        return tuple(sorted(self.items()))

    def __hash__(self):
        return hash(self.__tuplify__())

    def __lt__(self, odict):
        assert isinstance(odict, hashabledict)
        return self.__tuplify__() < odict.__tuplify__()

    def __gt__(self, odict):
        assert isinstance(odict, hashabledict)
        return self.__tuplify__() > odict.__tuplify__()

    def __le__(self, odict):
        assert isinstance(odict, hashabledict)
        return self.__tuplify__() <= odict.__tuplify__()

    def __ge__(self, odict):
        assert isinstance(odict, hashabledict)
        return self.__tuplify__() >= odict.__tuplify__()


# ______________________________________________________________________________
# Queues: Stack, FIFOQueue, PriorityQueue

# TODO: queue.PriorityQueue
# TODO: Priority queues may not belong here -- see treatment in search.py


class Queue:

    """Queue is an abstract class/interface. There are three types:
        Stack(): A Last In First Out Queue.
        FIFOQueue(): A First In First Out Queue.
        PriorityQueue(order, f): Queue in sorted order (default min-first).
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
        item in q       -- does q contain item?
    Note that isinstance(Stack(), Queue) is false, because we implement stacks
    as lists.  If Python ever gets interfaces, Queue will be an interface."""

    def __init__(self):
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.append(item)


def Stack():
    """Return an empty list, suitable as a Last-In-First-Out Queue."""
    return []


class FIFOQueue(Queue):

    """A First-In-First-Out Queue."""

    def __init__(self, maxlen=None, items=[]):
        self.queue = collections.deque(items, maxlen)

    def append(self, item):
        if not self.queue.maxlen or len(self.queue) < self.queue.maxlen:
            self.queue.append(item)
        else:
            raise Exception('FIFOQueue is full')

    def extend(self, items):
        if not self.queue.maxlen or len(self.queue) + len(items) <= self.queue.maxlen:
            self.queue.extend(items)
        else:
            raise Exception('FIFOQueue max length exceeded')

    def pop(self):
        if len(self.queue) > 0:
            return self.queue.popleft()
        else:
            raise Exception('FIFOQueue is empty')

    def __len__(self):
        return len(self.queue)

    def __contains__(self, item):
        return item in self.queue


class PriorityQueue(Queue):

    """A queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. If order is min, the item with minimum f(x) is
    returned first; if order is max, then it is the item with maximum f(x).
    Also supports dict-like lookup."""

    def __init__(self, order=min, f=lambda x: x):
        self.A = []
        self.order = order
        self.f = f

    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.A)

    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)


# ______________________________________________________________________________
# Useful Shorthands


class Bool(int):
    """Just like `bool`, except values display as 'T' and 'F' instead of 'True' and 'False'"""
    __str__ = __repr__ = lambda self: 'T' if self else 'F'


T = Bool(True)
F = Bool(False)



"""
Representations and Inference for Logic (Chapters 7-9, 12)

Covers both Propositional and First-Order Logic. First we have four
important data types:

    KB            Abstract class holds a knowledge base of logical expressions
    KB_Agent      Abstract class subclasses agents.Agent
    Expr          A logical expression, imported from utils.py
    substitution  Implemented as a dictionary of var:value pairs, {x:1, y:x}

Be careful: some functions take an Expr as argument, and some take a KB.

Logical expressions can be created with Expr or expr, imported from utils, TODO
or with expr, which adds the capability to write a string that uses
the connectives ==>, <==, <=>, or <=/=>. But be careful: these have the
operator precedence of commas; you may need to add parens to make precedence work.

Then we implement various functions for doing logical inference:

    pl_true          Evaluate a propositional logical sentence in a model
    tt_entails       Say if a statement is entailed by a KB
    pl_resolution    Do resolution on propositional sentences
    dpll_satisfiable See if a propositional sentence is satisfiable


And a few other functions:

    to_cnf           Convert to conjunctive normal form
    unify            Do unification of two FOL sentences
    diff, simp       Symbolic differentiation and simplification
"""

import heapq
import itertools
import random
from collections import defaultdict, Counter

import networkx as nx

class KB:
    """A knowledge base to which you can tell and ask sentences.
    To create a KB, first subclass this class and implement
    tell, ask_generator, and retract. Why ask_generator instead of ask?
    The book is a bit vague on what ask means --
    For a Propositional Logic KB, ask(P & Q) returns True or False, but for an
    FOL KB, something like ask(Brother(x, y)) might return many substitutions
    such as {x: Cain, y: Abel}, {x: Abel, y: Cain}, {x: George, y: Jeb}, etc.
    So ask_generator generates these one at a time, and ask either returns the
    first one or returns False."""

    def __init__(self, sentence=None):
        raise NotImplementedError

    def tell(self, sentence):
        """Add the sentence to the KB."""
        raise NotImplementedError

    def ask(self, query):
        """Return a substitution that makes the query true, or, failing that, return False."""
        return first(self.ask_generator(query), default=False)

    def ask_generator(self, query):
        """Yield all the substitutions that make query true."""
        raise NotImplementedError

    def retract(self, sentence):
        """Remove sentence from the KB."""
        raise NotImplementedError


class PropKB(KB):
    """A KB for propositional logic. Inefficient, with no indexing."""

    def __init__(self, sentence=None):
        self.clauses = []
        if sentence:
            self.tell(sentence)

    def tell(self, sentence):
        """Add the sentence's clauses to the KB."""
        self.clauses.extend(conjuncts(to_cnf(sentence)))

    def ask_generator(self, query):
        """Yield the empty substitution {} if KB entails query; else no results."""
        if tt_entails(Expr('&', *self.clauses), query):
            yield {}

    def ask_if_true(self, query):
        """Return True if the KB entails query, else return False."""
        for _ in self.ask_generator(query):
            return True
        return False

    def retract(self, sentence):
        """Remove the sentence's clauses from the KB."""
        for c in conjuncts(to_cnf(sentence)):
            if c in self.clauses:
                self.clauses.remove(c)


# ______________________________________________________________________________



def tt_entails(kb, alpha):
    """
    [Figure 7.10]
    Does kb entail the sentence alpha? Use truth tables. For propositional
    kb's and sentences. Note that the 'kb' should be an Expr which is a
    conjunction of clauses.
    >>> tt_entails(expr('P & Q'), expr('Q'))
    True
    """
    assert not variables(alpha)
    symbols = list(prop_symbols(kb & alpha))
    return tt_check_all(kb, alpha, symbols, {})

def is_variable(x):
    """A variable is an Expr with no args and a lowercase symbol as the op."""
    return isinstance(x, Expr) and not x.args and x.op[0].islower()



def variables(s):
    """Return a set of the variables in expression s.
    >>> variables(expr('F(x, x) & G(x, y) & H(y, z) & R(A, z, 2)')) == {x, y, z}
    True
    """
    return {x for x in subexpressions(s) if is_variable(x)}

def tt_check_all(kb, alpha, symbols, model):
    """Auxiliary routine to implement tt_entails."""
    if not symbols:
        if pl_true(kb, model):
            result = pl_true(alpha, model)
            assert result in (True, False)
            return result
        else:
            return True
    else:
        P, rest = symbols[0], symbols[1:]
        return (tt_check_all(kb, alpha, rest, extend(model, P, True)) and
                tt_check_all(kb, alpha, rest, extend(model, P, False)))


def extend(s, var, val):
    """Copy dict s and extend it by setting var to val; return copy."""
    return {**s, var: val}


def prop_symbols(x):
    """Return the set of all propositional symbols in x."""
    if not isinstance(x, Expr):
        return set()
    elif is_prop_symbol(x.op):
        return {x}
    else:
        return {symbol for arg in x.args for symbol in prop_symbols(arg)}


def constant_symbols(x):
    """Return the set of all constant symbols in x."""
    if not isinstance(x, Expr):
        return set()
    elif is_prop_symbol(x.op) and not x.args:
        return {x}
    else:
        return {symbol for arg in x.args for symbol in constant_symbols(arg)}


def predicate_symbols(x):
    """Return a set of (symbol_name, arity) in x.
    All symbols (even functional) with arity > 0 are considered."""
    if not isinstance(x, Expr) or not x.args:
        return set()
    pred_set = {(x.op, len(x.args))} if is_prop_symbol(x.op) else set()
    pred_set.update({symbol for arg in x.args for symbol in predicate_symbols(arg)})
    return pred_set


def tt_true(s):
    """Is a propositional sentence a tautology?
    >>> tt_true('P | ~P')
    True
    """
    s = expr(s)
    return tt_entails(True, s)


def pl_true(exp, model={}):
    """Return True if the propositional logic expression is true in the model,
    and False if it is false. If the model does not specify the value for
    every proposition, this may return None to indicate 'not obvious';
    this may happen even when the expression is tautological.
    >>> pl_true(P, {}) is None
    True
    """
    if exp in (True, False):
        return exp
    op, args = exp.op, exp.args
    if is_prop_symbol(op):
        return model.get(exp)
    elif op == '~':
        p = pl_true(args[0], model)
        if p is None:
            return None
        else:
            return not p
    elif op == '|':
        result = False
        for arg in args:
            p = pl_true(arg, model)
            if p is True:
                return True
            if p is None:
                result = None
        return result
    elif op == '&':
        result = True
        for arg in args:
            p = pl_true(arg, model)
            if p is False:
                return False
            if p is None:
                result = None
        return result
    p, q = args
    if op == '==>':
        return pl_true(~p | q, model)
    elif op == '<==':
        return pl_true(p | ~q, model)
    pt = pl_true(p, model)
    if pt is None:
        return None
    qt = pl_true(q, model)
    if qt is None:
        return None
    if op == '<=>':
        return pt == qt
    elif op == '^':  # xor or 'not equivalent'
        return pt != qt
    else:
        raise ValueError('Illegal operator in logic expression' + str(exp))


# ______________________________________________________________________________

# Convert to Conjunctive Normal Form (CNF)


def to_cnf(s):
    """
    [Page 253]
    Convert a propositional logical sentence to conjunctive normal form.
    That is, to the form ((A | ~B | ...) & (B | C | ...) & ...)
    >>> to_cnf('~(B | C)')
    (~B & ~C)
    """
    s = expr(s)
    if isinstance(s, str):
        s = expr(s)
    s = eliminate_implications(s)  # Steps 1, 2 from p. 253
    s = move_not_inwards(s)  # Step 3
    return distribute_and_over_or(s)  # Step 4



def is_symbol(s):
    """A string s is a symbol if it starts with an alphabetic char.
    >>> is_symbol('R2D2')
    True
    """
    return isinstance(s, str) and s[:1].isalpha()


def is_prop_symbol(s):
    """A proposition logic symbol is an initial-uppercase string.
    >>> is_prop_symbol('exe')
    False
    """
    return is_symbol(s) and s[0].isupper()

def eliminate_implications(s):
    """Change implications into equivalent form with only &, |, and ~ as logical operators."""
    s = expr(s)
    if not s.args or is_symbol(s.op):
        return s  # Atoms are unchanged.
    args = list(map(eliminate_implications, s.args))
    a, b = args[0], args[-1]
    if s.op == '==>':
        return b | ~a
    elif s.op == '<==':
        return a | ~b
    elif s.op == '<=>':
        return (a | ~b) & (b | ~a)
    elif s.op == '^':
        assert len(args) == 2  # TODO: relax this restriction
        return (a & ~b) | (~a & b)
    else:
        assert s.op in ('&', '|', '~')
        return Expr(s.op, *args)


def move_not_inwards(s):
    """Rewrite sentence s by moving negation sign inward.
    >>> move_not_inwards(~(A | B))
    (~A & ~B)
    """
    s = expr(s)
    if s.op == '~':
        def NOT(b):
            return move_not_inwards(~b)

        a = s.args[0]
        if a.op == '~':
            return move_not_inwards(a.args[0])  # ~~A ==> A
        if a.op == '&':
            return associate('|', list(map(NOT, a.args)))
        if a.op == '|':
            return associate('&', list(map(NOT, a.args)))
        return s
    elif is_symbol(s.op) or not s.args:
        return s
    else:
        return Expr(s.op, *list(map(move_not_inwards, s.args)))


def distribute_and_over_or(s):
    """Given a sentence s consisting of conjunctions and disjunctions
    of literals, return an equivalent sentence in CNF.
    >>> distribute_and_over_or((A & B) | C)
    ((A | C) & (B | C))
    """
    s = expr(s)
    if s.op == '|':
        s = associate('|', s.args)
        if s.op != '|':
            return distribute_and_over_or(s)
        if len(s.args) == 0:
            return False
        if len(s.args) == 1:
            return distribute_and_over_or(s.args[0])
        conj = first(arg for arg in s.args if arg.op == '&')
        if not conj:
            return s
        others = [a for a in s.args if a is not conj]
        rest = associate('|', others)
        return associate('&', [distribute_and_over_or(c | rest)
                               for c in conj.args])
    elif s.op == '&':
        return associate('&', list(map(distribute_and_over_or, s.args)))
    else:
        return s


def associate(op, args):
    """Given an associative op, return an expression with the same
    meaning as Expr(op, *args), but flattened -- that is, with nested
    instances of the same op promoted to the top level.
    >>> associate('&', [(A&B),(B|C),(B&C)])
    (A & B & (B | C) & B & C)
    >>> associate('|', [A|(B|(C|(A&B)))])
    (A | B | C | (A & B))
    """
    args = dissociate(op, args)
    if len(args) == 0:
        return _op_identity[op]
    elif len(args) == 1:
        return args[0]
    else:
        return Expr(op, *args)


_op_identity = {'&': True, '|': False, '+': 0, '*': 1}


def dissociate(op, args):
    """Given an associative op, return a flattened list result such
    that Expr(op, *result) means the same as Expr(op, *args).
    >>> dissociate('&', [A & B])
    [A, B]
    """
    result = []

    def collect(subargs):
        for arg in subargs:
            if arg.op == op:
                collect(arg.args)
            else:
                result.append(arg)

    collect(args)
    return result


def conjuncts(s):
    """Return a list of the conjuncts in the sentence s.
    >>> conjuncts(A & B)
    [A, B]
    >>> conjuncts(A | B)
    [(A | B)]
    """
    return dissociate('&', [s])


def disjuncts(s):
    """Return a list of the disjuncts in the sentence s.
    >>> disjuncts(A | B)
    [A, B]
    >>> disjuncts(A & B)
    [(A & B)]
    """
    return dissociate('|', [s])


# ______________________________________________________________________________


def pl_resolution(kb, alpha):
    """
    [Figure 7.12]
    Propositional-logic resolution: say if alpha follows from KB.
    >>> pl_resolution(horn_clauses_KB, A)
    True
    """
    clauses = kb.clauses + conjuncts(to_cnf(~alpha))
    new = set()
    while True:
        n = len(clauses)
        pairs = [(clauses[i], clauses[j])
                 for i in range(n) for j in range(i + 1, n)]
        for (ci, cj) in pairs:
            resolvents = pl_resolve(ci, cj)
            if False in resolvents:
                return True
            new = new.union(set(resolvents))
        if new.issubset(set(clauses)):
            return False
        for c in new:
            if c not in clauses:
                clauses.append(c)


def pl_resolve(ci, cj):
    """Return all clauses that can be obtained by resolving clauses ci and cj."""
    clauses = []
    for di in disjuncts(ci):
        for dj in disjuncts(cj):
            if di == ~dj or ~di == dj:
                clauses.append(associate('|', unique(remove_all(di, disjuncts(ci)) + remove_all(dj, disjuncts(cj)))))
    return clauses


# ______________________________________________________________________________

def is_definite_clause(s):
    """Returns True for exprs s of the form A & B & ... & C ==> D,
    where all literals are positive. In clause form, this is
    ~A | ~B | ... | ~C | D, where exactly one clause is positive.
    >>> is_definite_clause(expr('Farmer(Mac)'))
    True
    """
    if is_symbol(s.op):
        return True
    elif s.op == '==>':
        antecedent, consequent = s.args
        return is_symbol(consequent.op) and all(is_symbol(arg.op) for arg in conjuncts(antecedent))
    else:
        return False
    
class PropDefiniteKB(PropKB):
    """A KB of propositional definite clauses."""

    def tell(self, sentence):
        """Add a definite clause to this KB."""
        assert is_definite_clause(sentence), "Must be definite clause"
        self.clauses.append(sentence)

    def ask_generator(self, query):
        """Yield the empty substitution if KB implies query; else nothing."""
        if pl_fc_entails(self.clauses, query):
            yield {}

    def retract(self, sentence):
        self.clauses.remove(sentence)

    def clauses_with_premise(self, p):
        """Return a list of the clauses in KB that have p in their premise.
        This could be cached away for O(1) speed, but we'll recompute it."""
        return [c for c in self.clauses if c.op == '==>' and p in conjuncts(c.args[0])]


def pl_fc_entails(kb, q):
    """
    [Figure 7.15]
    Use forward chaining to see if a PropDefiniteKB entails symbol q.
    >>> pl_fc_entails(horn_clauses_KB, expr('Q'))
    True
    """
    count = {c: len(conjuncts(c.args[0])) for c in kb.clauses if c.op == '==>'}
    inferred = defaultdict(bool)
    agenda = [s for s in kb.clauses if is_prop_symbol(s.op)]
    while agenda:
        p = agenda.pop()
        if p == q:
            return True
        if not inferred[p]:
            inferred[p] = True
            for c in kb.clauses_with_premise(p):
                count[c] -= 1
                if count[c] == 0:
                    agenda.append(c.args[1])
    return False


"""
[Figure 7.13]
Simple inference in a wumpus world example
"""
wumpus_world_inference = expr('(B11 <=> (P12 | P21))  &  ~B11')

"""
[Figure 7.16]
Propositional Logic Forward Chaining example
"""
horn_clauses_KB = PropDefiniteKB()
for clause in ['P ==> Q',
               '(L & M) ==> P',
               '(B & L) ==> M',
               '(A & P) ==> L',
               '(A & B) ==> L',
               'A', 'B']:
    horn_clauses_KB.tell(expr(clause))

"""
Definite clauses KB example
"""
definite_clauses_KB = PropDefiniteKB()
for clause in ['(B & F) ==> E',
               '(A & E & F) ==> G',
               '(B & C) ==> F',
               '(A & B) ==> D',
               '(E & F) ==> H',
               '(H & I) ==>J',
               'A', 'B', 'C']:
    definite_clauses_KB.tell(expr(clause))


# ______________________________________________________________________________
# Heuristics for SAT Solvers


def no_branching_heuristic(symbols, clauses):
    return first(symbols), True


def min_clauses(clauses):
    min_len = min(map(lambda c: len(c.args), clauses), default=2)
    return filter(lambda c: len(c.args) == (min_len if min_len > 1 else 2), clauses)


def moms(symbols, clauses):
    """
    MOMS (Maximum Occurrence in clauses of Minimum Size) heuristic
    Returns the literal with the most occurrences in all clauses of minimum size
    """
    scores = Counter(l for c in min_clauses(clauses) for l in prop_symbols(c))
    return max(symbols, key=lambda symbol: scores[symbol]), True


def momsf(symbols, clauses, k=0):
    """
    MOMS alternative heuristic
    If f(x) the number of occurrences of the variable x in clauses with minimum size,
    we choose the variable maximizing [f(x) + f(-x)] * 2^k + f(x) * f(-x)
    Returns x if f(x) >= f(-x) otherwise -x
    """
    scores = Counter(l for c in min_clauses(clauses) for l in disjuncts(c))
    P = max(symbols,
            key=lambda symbol: (scores[symbol] + scores[~symbol]) * pow(2, k) + scores[symbol] * scores[~symbol])
    return P, True if scores[P] >= scores[~P] else False


def posit(symbols, clauses):
    """
    Freeman's POSIT version of MOMs
    Counts the positive x and negative x for each variable x in clauses with minimum size
    Returns x if f(x) >= f(-x) otherwise -x
    """
    scores = Counter(l for c in min_clauses(clauses) for l in disjuncts(c))
    P = max(symbols, key=lambda symbol: scores[symbol] + scores[~symbol])
    return P, True if scores[P] >= scores[~P] else False


def zm(symbols, clauses):
    """
    Zabih and McAllester's version of MOMs
    Counts the negative occurrences only of each variable x in clauses with minimum size
    """
    scores = Counter(l for c in min_clauses(clauses) for l in disjuncts(c) if l.op == '~')
    return max(symbols, key=lambda symbol: scores[~symbol]), True


def dlis(symbols, clauses):
    """
    DLIS (Dynamic Largest Individual Sum) heuristic
    Choose the variable and value that satisfies the maximum number of unsatisfied clauses
    Like DLCS but we only consider the literal (thus Cp and Cn are individual)
    """
    scores = Counter(l for c in clauses for l in disjuncts(c))
    P = max(symbols, key=lambda symbol: scores[symbol])
    return P, True if scores[P] >= scores[~P] else False


def dlcs(symbols, clauses):
    """
    DLCS (Dynamic Largest Combined Sum) heuristic
    Cp the number of clauses containing literal x
    Cn the number of clauses containing literal -x
    Here we select the variable maximizing Cp + Cn
    Returns x if Cp >= Cn otherwise -x
    """
    scores = Counter(l for c in clauses for l in disjuncts(c))
    P = max(symbols, key=lambda symbol: scores[symbol] + scores[~symbol])
    return P, True if scores[P] >= scores[~P] else False


def jw(symbols, clauses):
    """
    Jeroslow-Wang heuristic
    For each literal compute J(l) = \sum{l in clause c} 2^{-|c|}
    Return the literal maximizing J
    """
    scores = Counter()
    for c in clauses:
        for l in prop_symbols(c):
            scores[l] += pow(2, -len(c.args))
    return max(symbols, key=lambda symbol: scores[symbol]), True


def jw2(symbols, clauses):
    """
    Two Sided Jeroslow-Wang heuristic
    Compute J(l) also counts the negation of l = J(x) + J(-x)
    Returns x if J(x) >= J(-x) otherwise -x
    """
    scores = Counter()
    for c in clauses:
        for l in disjuncts(c):
            scores[l] += pow(2, -len(c.args))
    P = max(symbols, key=lambda symbol: scores[symbol] + scores[~symbol])
    return P, True if scores[P] >= scores[~P] else False


# ______________________________________________________________________________
# DPLL-Satisfiable [Figure 7.17]


def dpll_satisfiable(s, branching_heuristic=no_branching_heuristic):
    """Check satisfiability of a propositional sentence.
    This differs from the book code in two ways: (1) it returns a model
    rather than True when it succeeds; this is more useful. (2) The
    function find_pure_symbol is passed a list of unknown clauses, rather
    than a list of all clauses and the model; this is more efficient.
    >>> dpll_satisfiable(A |'<=>'| B) == {A: True, B: True}
    True
    """
    return dpll(conjuncts(to_cnf(s)), prop_symbols(s), {}, branching_heuristic)


def dpll(clauses, symbols, model, branching_heuristic=no_branching_heuristic):
    """See if the clauses are true in a partial model."""
    unknown_clauses = []  # clauses with an unknown truth value
    for c in clauses:
        val = pl_true(c, model)
        if val is False:
            return False
        if val is None:
            unknown_clauses.append(c)
    if not unknown_clauses:
        return model
    P, value = find_pure_symbol(symbols, unknown_clauses)
    if P:
        return dpll(clauses, remove_all(P, symbols), extend(model, P, value), branching_heuristic)
    P, value = find_unit_clause(clauses, model)
    if P:
        return dpll(clauses, remove_all(P, symbols), extend(model, P, value), branching_heuristic)
    P, value = branching_heuristic(symbols, unknown_clauses)
    return (dpll(clauses, remove_all(P, symbols), extend(model, P, value), branching_heuristic) or
            dpll(clauses, remove_all(P, symbols), extend(model, P, not value), branching_heuristic))


def find_pure_symbol(symbols, clauses):
    """Find a symbol and its value if it appears only as a positive literal
    (or only as a negative) in clauses.
    >>> find_pure_symbol([A, B, C], [A|~B,~B|~C,C|A])
    (A, True)
    """
    for s in symbols:
        found_pos, found_neg = False, False
        for c in clauses:
            if not found_pos and s in disjuncts(c):
                found_pos = True
            if not found_neg and ~s in disjuncts(c):
                found_neg = True
        if found_pos != found_neg:
            return s, found_pos
    return None, None


def find_unit_clause(clauses, model):
    """Find a forced assignment if possible from a clause with only 1
    variable not bound in the model.
    >>> find_unit_clause([A|B|C, B|~C, ~A|~B], {A:True})
    (B, False)
    """
    for clause in clauses:
        P, value = unit_clause_assign(clause, model)
        if P:
            return P, value
    return None, None


def unit_clause_assign(clause, model):
    """Return a single variable/value pair that makes clause true in
    the model, if possible.
    >>> unit_clause_assign(A|B|C, {A:True})
    (None, None)
    >>> unit_clause_assign(B|~C, {A:True})
    (None, None)
    >>> unit_clause_assign(~A|~B, {A:True})
    (B, False)
    """
    P, value = None, None
    for literal in disjuncts(clause):
        sym, positive = inspect_literal(literal)
        if sym in model:
            if model[sym] == positive:
                return None, None  # clause already True
        elif P:
            return None, None  # more than 1 unbound variable
        else:
            P, value = sym, positive
    return P, value


def inspect_literal(literal):
    """The symbol in this literal, and the value it should take to
    make the literal true.
    >>> inspect_literal(P)
    (P, True)
    >>> inspect_literal(~P)
    (P, False)
    """
    if literal.op == '~':
        return literal.args[0], False
    else:
        return literal, True


# ______________________________________________________________________________



# Expr functions for WumpusKB and HybridWumpusAgent

def facing_east(time):
    return Expr('FacingEast', time)


def facing_west(time):
    return Expr('FacingWest', time)


def facing_north(time):
    return Expr('FacingNorth', time)


def facing_south(time):
    return Expr('FacingSouth', time)


def wumpus(x, y):
    return Expr('W', x, y)


def pit(x, y):
    return Expr('P', x, y)


def breeze(x, y):
    return Expr('B', x, y)


def stench(x, y):
    return Expr('S', x, y)


def wumpus_alive(time):
    return Expr('WumpusAlive', time)


def have_arrow(time):
    return Expr('HaveArrow', time)


def percept_stench(time):
    return Expr('Stench', time)


def percept_breeze(time):
    return Expr('Breeze', time)


def percept_glitter(time):
    return Expr('Glitter', time)


def percept_bump(time):
    return Expr('Bump', time)


def percept_scream(time):
    return Expr('Scream', time)


def move_forward(time):
    return Expr('Forward', time)


def shoot(time):
    return Expr('Shoot', time)


def turn_left(time):
    return Expr('TurnLeft', time)


def turn_right(time):
    return Expr('TurnRight', time)


def ok_to_move(x, y, time):
    return Expr('OK', x, y, time)


def location(x, y, time=None):
    if time is None:
        return Expr('L', x, y)
    else:
        return Expr('L', x, y, time)


# Symbols

def implies(lhs, rhs):
    return Expr('==>', lhs, rhs)


def equiv(lhs, rhs):
    return Expr('<=>', lhs, rhs)


# Helper Function

def new_disjunction(sentences):
    t = sentences[0]
    for i in range(1, len(sentences)):
        t |= sentences[i]
    return t


# ______________________________________________________________________________


class WumpusKB(PropKB):
    """
    Create a Knowledge Base that contains the a temporal "Wumpus physics" and temporal rules with time zero.
    """

    def __init__(self, dimrow):
        super().__init__()
        self.dimrow = dimrow
        self.tell(~wumpus(1, 1))
        self.tell(~pit(1, 1))

        for y in range(1, dimrow + 1):
            for x in range(1, dimrow + 1):

                pits_in = list()
                wumpus_in = list()

                if x > 1:  # West room exists
                    pits_in.append(pit(x - 1, y))
                    wumpus_in.append(wumpus(x - 1, y))

                if y < dimrow:  # North room exists
                    pits_in.append(pit(x, y + 1))
                    wumpus_in.append(wumpus(x, y + 1))

                if x < dimrow:  # East room exists
                    pits_in.append(pit(x + 1, y))
                    wumpus_in.append(wumpus(x + 1, y))

                if y > 1:  # South room exists
                    pits_in.append(pit(x, y - 1))
                    wumpus_in.append(wumpus(x, y - 1))

                self.tell(equiv(breeze(x, y), new_disjunction(pits_in)))
                self.tell(equiv(stench(x, y), new_disjunction(wumpus_in)))

        # Rule that describes existence of at least one Wumpus
        wumpus_at_least = list()
        for x in range(1, dimrow + 1):
            for y in range(1, dimrow + 1):
                wumpus_at_least.append(wumpus(x, y))

        self.tell(new_disjunction(wumpus_at_least))

        # Rule that describes existence of at most one Wumpus
        for i in range(1, dimrow + 1):
            for j in range(1, dimrow + 1):
                for u in range(1, dimrow + 1):
                    for v in range(1, dimrow + 1):
                        if i != u or j != v:
                            self.tell(~wumpus(i, j) | ~wumpus(u, v))

        # Temporal rules at time zero
        self.tell(location(1, 1, 0))
        for i in range(1, dimrow + 1):
            for j in range(1, dimrow + 1):
                self.tell(implies(location(i, j, 0), equiv(percept_breeze(0), breeze(i, j))))
                self.tell(implies(location(i, j, 0), equiv(percept_stench(0), stench(i, j))))
                if i != 1 or j != 1:
                    self.tell(~location(i, j, 0))

        self.tell(wumpus_alive(0))
        self.tell(have_arrow(0))
        self.tell(facing_east(0))
        self.tell(~facing_north(0))
        self.tell(~facing_south(0))
        self.tell(~facing_west(0))

    def make_action_sentence(self, action, time):
        actions = [move_forward(time), shoot(time), turn_left(time), turn_right(time)]

        for a in actions:
            if action is a:
                self.tell(action)
            else:
                self.tell(~a)

    def make_percept_sentence(self, percept, time):
        # Glitter, Bump, Stench, Breeze, Scream
        flags = [0, 0, 0, 0, 0]

        # Things perceived
        if isinstance(percept, Glitter):
            flags[0] = 1
            self.tell(percept_glitter(time))
        elif isinstance(percept, Bump):
            flags[1] = 1
            self.tell(percept_bump(time))
        elif isinstance(percept, Stench):
            flags[2] = 1
            self.tell(percept_stench(time))
        elif isinstance(percept, Breeze):
            flags[3] = 1
            self.tell(percept_breeze(time))
        elif isinstance(percept, Scream):
            flags[4] = 1
            self.tell(percept_scream(time))

        # Things not perceived
        for i in range(len(flags)):
            if flags[i] == 0:
                if i == 0:
                    self.tell(~percept_glitter(time))
                elif i == 1:
                    self.tell(~percept_bump(time))
                elif i == 2:
                    self.tell(~percept_stench(time))
                elif i == 3:
                    self.tell(~percept_breeze(time))
                elif i == 4:
                    self.tell(~percept_scream(time))

    def add_temporal_sentences(self, time):
        if time == 0:
            return
        t = time - 1

        # current location rules
        for i in range(1, self.dimrow + 1):
            for j in range(1, self.dimrow + 1):
                self.tell(implies(location(i, j, time), equiv(percept_breeze(time), breeze(i, j))))
                self.tell(implies(location(i, j, time), equiv(percept_stench(time), stench(i, j))))
                s = list()
                s.append(equiv(location(i, j, time), location(i, j, time) & ~move_forward(time) | percept_bump(time)))
                if i != 1:
                    s.append(location(i - 1, j, t) & facing_east(t) & move_forward(t))
                if i != self.dimrow:
                    s.append(location(i + 1, j, t) & facing_west(t) & move_forward(t))
                if j != 1:
                    s.append(location(i, j - 1, t) & facing_north(t) & move_forward(t))
                if j != self.dimrow:
                    s.append(location(i, j + 1, t) & facing_south(t) & move_forward(t))

                # add sentence about location i,j
                self.tell(new_disjunction(s))

                # add sentence about safety of location i,j
                self.tell(equiv(ok_to_move(i, j, time), ~pit(i, j) & ~wumpus(i, j) & wumpus_alive(time)))

        # Rules about current orientation

        a = facing_north(t) & turn_right(t)
        b = facing_south(t) & turn_left(t)
        c = facing_east(t) & ~turn_left(t) & ~turn_right(t)
        s = equiv(facing_east(time), a | b | c)
        self.tell(s)

        a = facing_north(t) & turn_left(t)
        b = facing_south(t) & turn_right(t)
        c = facing_west(t) & ~turn_left(t) & ~turn_right(t)
        s = equiv(facing_west(time), a | b | c)
        self.tell(s)

        a = facing_east(t) & turn_left(t)
        b = facing_west(t) & turn_right(t)
        c = facing_north(t) & ~turn_left(t) & ~turn_right(t)
        s = equiv(facing_north(time), a | b | c)
        self.tell(s)

        a = facing_west(t) & turn_left(t)
        b = facing_east(t) & turn_right(t)
        c = facing_south(t) & ~turn_left(t) & ~turn_right(t)
        s = equiv(facing_south(time), a | b | c)
        self.tell(s)

        # Rules about last action
        self.tell(equiv(move_forward(t), ~turn_right(t) & ~turn_left(t)))

        # Rule about the arrow
        self.tell(equiv(have_arrow(time), have_arrow(t) & ~shoot(t)))

        # Rule about Wumpus (dead or alive)
        self.tell(equiv(wumpus_alive(time), wumpus_alive(t) & ~percept_scream(time)))

    def ask_if_true(self, query):
        return pl_resolution(self, query)
