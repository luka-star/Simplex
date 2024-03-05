import timeit
import scipy.optimize
import numpy as np
from fractions import Fraction
from enum import Enum
import matplotlib.pyplot as plt




def example1():
    return (
        np.array([5, 4, 3]),
        np.array([[2, 3, 1], [4, 1, 2], [3, 4, 2]]),
        np.array([5, 11, 8]),
    )


def example2():
    return (
        np.array([-2, -1]),
        np.array([[-1, 1], [-1, -2], [0, 1]]),
        np.array([-1, -2, 1]),
    )


def integer_pivoting_example():
    return np.array([5, 2]), np.array([[3, 1], [2, 5]]), np.array([7, 5])


def exercise2_5():
    return (
        np.array([1, 3]),
        np.array([[-1, -1], [-1, 1], [1, 2]]),
        np.array([-3, -1, 4]),
    )


def exercise2_6():
    return (
        np.array([1, 3]),
        np.array([[-1, -1], [-1, 1], [1, 2]]),
        np.array([-3, -1, 2]),
    )


def exercise2_7():
    return (
        np.array([1, 3]),
        np.array([[-1, -1], [-1, 1], [-1, 2]]),
        np.array([-3, -1, 2]),
    )


def cycleexample():
    return (
        np.array([1, -2, 0, -2]),
        np.array([[0.5, -3.5, -2, 4], [0.5, -1, -0.5, 0.5], [1, 0, 0, 0]]),
        np.array([0, 0, 1]),
    )


def auxexample():
    return (
        np.array([-2, -1]),
        np.array([[-1, 1], [-1, -2], [0, 1]]),
        np.array([-1, -2, 1]),
    )


def random_lp(n, m, sigma=10, seed=0):
    np.random.seed(seed)
    return (
        np.round(sigma * np.random.randn(n)),
        np.round(sigma * np.random.randn(m, n)),
        np.round(sigma * np.abs(np.random.randn(m))),
    )


class Dictionary:
    # Simplex dictionary as defined by Vanderbei
    #
    # 'C' is a (m+1)x(n+1) NumPy array that stores all the coefficients
    # of the dictionary.
    #
    # 'dtype' is the type of the entries of the dictionary. It is
    # supposed to be one of the native (full precision) Python types
    # 'int' or 'Fraction' or any Numpy type such as 'np.float64'.
    #
    # dtype 'int' is used for integer pivoting. Here an additional
    # variables 'lastpivot' is used. 'lastpivot' is the negative pivot
    # coefficient of the previous pivot operation. Dividing all
    # entries of the integer dictionary by 'lastpivot' results in the
    # normal dictionary.
    #
    # Variables are indexed from 0 to n+m. Variable 0 is the objective
    # z. Variables 1 to n are the original variables. Variables n+1 to
    # n+m are the slack variables. An exception is when creating an
    # auxillary dictionary where variable n+1 is the auxillary
    # variable (named x0) and variables n+2 to n+m+1 are the slack
    # variables (still names x{n+1} to x{n+m}).
    #
    # 'B' and 'N' are arrays that contain the *indices* of the basic and
    # nonbasic variables.
    #
    # 'varnames' is an array of the names of the variables.

    def __init__(self, c, A, b, dtype=Fraction):
        # Initializes the dictionary based on linear program in
        # standard form given by vectors and matrices 'c','A','b'.
        # Dimensions are inferred from 'A'
        #
        # If 'c' is None it generates the auxillary dictionary for the
        # use in the standard two-phase simplex algorithm
        #
        # Every entry of the input is individually converted to the
        # given dtype.
        m, n = A.shape
        self.dtype = dtype
        if dtype == int:
            self.lastpivot = 1
        if dtype in [int, Fraction]:
            dtype = object
            if c is not None:
                c = np.array(c, dtype=object)
            A = np.array(A, dtype=object)
            b = np.array(b, dtype=object)
        self.C = np.empty([m + 1, n + 1 + (c is None)], dtype=dtype)
        self.C[0, 0] = self.dtype(0)
        if c is None:
            self.C[0, 1:] = self.dtype(0)
            self.C[0, n + 1] = self.dtype(-1)
            self.C[1:, n + 1] = self.dtype(1)
        else:
            for j in range(0, n):
                self.C[0, j + 1] = self.dtype(c[j])
        for i in range(0, m):
            self.C[i + 1, 0] = self.dtype(b[i])
            for j in range(0, n):
                self.C[i + 1, j + 1] = self.dtype(-A[i, j])

        self.N = np.array(range(1, n + 1 + (c is None)))
        self.B = np.array(range(n + 1 + (c is None), n + 1 + (c is None) + m))
        self.varnames = np.empty(n + 1 + (c is None) + m, dtype=object)
        self.varnames[0] = "z"
        for i in range(1, n + 1):
            self.varnames[i] = "x{}".format(i)
        if c is None:
            self.varnames[n + 1] = "x0"
        for i in range(n + 1, n + m + 1):
            self.varnames[i + (c is None)] = "x{}".format(i)

    def __str__(self):
        # String representation of the dictionary in equation form as
        # used in Vanderbei.
        m, n = self.C.shape
        varlen = len(max(self.varnames, key=len))
        coeflen = 0
        for i in range(0, m):
            coeflen = max(coeflen, len(str(self.C[i, 0])))
            for j in range(1, n):
                coeflen = max(coeflen, len(str(abs(self.C[i, j]))))
        tmp = []
        if self.dtype == int and self.lastpivot != 1:
            tmp.append(str(self.lastpivot))
            tmp.append("*")
        tmp.append("{} = ".format(self.varnames[0]).rjust(varlen + 3))
        tmp.append(str(self.C[0, 0]).rjust(coeflen))
        for j in range(0, n - 1):
            tmp.append(" + " if self.C[0, j + 1] > 0 else " - ")
            tmp.append(str(abs(self.C[0, j + 1])).rjust(coeflen))
            tmp.append("*")
            tmp.append("{}".format(self.varnames[self.N[j]]).rjust(varlen))
        for i in range(0, m - 1):
            tmp.append("\n")
            if self.dtype == int and self.lastpivot != 1:
                tmp.append(str(self.lastpivot))
                tmp.append("*")
            tmp.append("{} = ".format(self.varnames[self.B[i]]).rjust(varlen + 3))
            tmp.append(str(self.C[i + 1, 0]).rjust(coeflen))
            for j in range(0, n - 1):
                tmp.append(" + " if self.C[i + 1, j + 1] > 0 else " - ")
                tmp.append(str(abs(self.C[i + 1, j + 1])).rjust(coeflen))
                tmp.append("*")
                tmp.append("{}".format(self.varnames[self.N[j]]).rjust(varlen))
        return "".join(tmp)

    def basic_solution(self):
        # Extracts the basic solution defined by a dictionary D
        m, n = self.C.shape
        if self.dtype == int:
            x_dtype = Fraction
        else:
            x_dtype = self.dtype
        x = np.empty(n - 1, x_dtype)
        x[:] = x_dtype(0)
        for i in range(0, m - 1):
            if self.B[i] < n:
                if self.dtype == int:
                    x[self.B[i] - 1] = Fraction(self.C[i + 1, 0], self.lastpivot)
                else:
                    x[self.B[i] - 1] = self.C[i + 1, 0]
        return x

    def value(self):
        # Extracts the value of the basic solution defined by a dictionary D
        if self.dtype == int:
            return Fraction(self.C[0, 0], self.lastpivot)
        else:
            return self.C[0, 0]

    def pivot(self, k, l):
        # Pivot Dictionary with N[k] entering and B[l] leaving
        # Performs integer pivoting if self.dtype==int
        # l is pivot row
        # k is pivot column
        # save pivot coefficient

        # print("Pivoting ", k, l)
        a = self.C[l + 1, k + 1]

        # Update all elements of the matrix except the pivot row and pivot column
        for i in range(self.C.shape[0]):
            for j in range(self.C.shape[1]):
                if i != l + 1 and j != k + 1:
                    b = self.C[i, k + 1]
                    c = self.C[l + 1, j]
                    self.C[i, j] = self.C[i, j] - (c * b) / a

        # Perform row operations for the pivot row
        self.C[l + 2 :, k + 1] /= a
        self.C[: l + 1, k + 1] /= a
        self.C[l + 1, : k + 1] /= -a
        self.C[l + 1, k + 2 :] /= -a
        self.C[l + 1, k + 1] = 1 / a

        # Swap entering and leaving variables
        self.N[k], self.B[l] = self.B[l], self.N[k]


class LPResult(Enum):
    OPTIMAL = 1
    INFEASIBLE = 2
    UNBOUNDED = 3


# Find the first variable in the objective function and return its index, otherwise return None
def find_entering(D, eps):
    coeffs = []
    for i, e in enumerate(D.N):
        coeffs.append((i, e))

    coeffs = sorted(coeffs, key=(lambda a: a[1]))
    for i in coeffs:
        if D.C[0, i[0]+1] > eps:
            return i[0]
    return None
    """ maxcoef = np.max(D.C[0,1:])  
    if maxcoef <= eps:
        return None
    else:
        return np.argmin(D.C[0, 1:])  """


# Choose leaving variable according to one-phase simplex
def find_leaving(D, k, eps):
    l = None
    min_ratio = np.inf
    min_indices = []
    allratio = []
    for i in range(1, D.C.shape[0]):
        if not (D.C[i, k + 1] <= eps and D.C[i, k + 1] >= -eps):
            ratio = D.C[i, 0] / D.C[i, k + 1]
            allratio.append(ratio)
            ratio = np.abs(ratio)
            if min_ratio > ratio:
                min_ratio = ratio
                min_indices = [i - 1]
            elif min_ratio == ratio:
                min_indices.append(i - 1)
        else:
            allratio.append(0)
    if min_indices:
        l = min(min_indices)

    if all(x < eps for x in D.C[:, k]) or all(x >= eps for x in D.C[:, k]) or all(x < eps for x in allratio):
        return None

    return l


def largest_coefficient(D, eps):
    # Assumes a feasible dictionary D and find entering and leaving
    # variables according to the Largest Coefficient rule.
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    # Returns k and l such that
    # k is None if D is Optimal
    # Otherwise D.N[k] is entering variable
    # l is None if D is Unbounded
    # Otherwise D.B[l] is a leaving variable

    k = l = None

    # Find entering variable according to largest coefficient rule
    maxcoef = np.max(D.C[0, 1:])
    if maxcoef <= eps:
        k = None
    else:
        k = np.argmax(D.C[0, 1:])

        if np.all(D.C[1:, k] >= -eps):
            l = None

        # Find leaving variable
        min_ratio = np.inf
        for i in range(1, D.C.shape[0]):
            if D.C[i, k] > eps:
                ratio = -(D.C[i, 0] / D.C[i, k + 1])
                if ratio < min_ratio:
                    min_ratio = ratio
                    l = i - 1

    return k, l


def largest_increase(D, eps):
    # Assumes a feasible dictionary D and find entering and leaving
    # variables according to the Largest Increase rule.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    #
    # Returns k and l such that
    # k is None if D is Optimal
    # Otherwise D.N[k] is entering variable
    # l is None if D is Unbounded
    # Otherwise D.B[l] is a leaving variable
    k = l = None
    
    


    return k, l

def helper(D, eps=0):
    while(True):
        k = find_entering(D, eps)

        if k == None:
            return LPResult.OPTIMAL, D

        l = find_leaving(D, k, eps)

        if l == None:
            return LPResult.UNBOUNDED, D

        D.pivot(k, l)

# Implementing twophase Simplex using auxillary var.
def two_phase_solve(c, A, b, dtype=Fraction, eps=0):
    # Phase 1
    D = Dictionary(c, A, b, dtype)  # copy the objective function
    objfun = D.C[0, :]
    # print(objfun)
    auxD = Dictionary(None, A, b, dtype)  # create auxillary dictionary
    x0 = len(auxD.N)
    k = len(auxD.N) - 1
    l = np.argmin(auxD.C[1:, 0])
    if D.C[l + 1, 0] <= eps and D.C[l + 1, 0] >= -eps:
        print(D.C[l +1, 0])
        return LPResult.OPTIMAL, auxD
    auxD.pivot(k, l)
    print(auxD)
    auxRes, auxD = helper(auxD)

    if auxRes != LPResult.OPTIMAL:
        return auxRes, auxD

    if x0 in auxD.B:
        indexOfx0 = (np.where(auxD.B == x0))[0][0]
        # print(indexOfx0)
        if -auxD.C[indexOfx0 + 1, 0] < 0:
            print("w < 0 == -x_0 < 0, aka x_0 positive")
            return LPResult.INFEASIBLE, auxD
        auxD.pivot(0, indexOfx0)

    x0InN = (np.where(auxD.N == x0))[0][0]
    auxD.N = np.delete(auxD.N, x0InN)
    auxD.C = np.delete(auxD.C, x0InN + 1, 1)

    while True:
        all_ogs_are_basic = True
        for i in D.N:
            if not (i in auxD.B):
                all_ogs_are_basic = False
        if all_ogs_are_basic:  # if all original non-basic variables are basic
            indexes = []
            for i in range(len(auxD.B)):
                if auxD.B[i] in D.N:
                    indexes.append((i + 1, c[auxD.B[i] - 1]))
            for i, j in indexes:
                auxD.C[0, :] += auxD.C[i, :] * j
            break
        else:
            for i in D.N:
                if not (i in auxD.B):
                    index_of_n = np.where(auxD.N == i)[0][0]
                    leaving = np.where(auxD.B >= len(D.N) + 1)[0][0]
                    auxD.pivot(index_of_n, leaving)
    print(auxD)

    return lp_solve(auxD)


def lp_solve(D, eps=0, pivotrule=lambda D: bland(D, eps=0), verbose=False):
    while True:
        if not (np.all(D.C[1:, 0] >= -eps)) and np.all(D.C[0, 1:] < -eps):
            return LPResult.INFEASIBLE, D

        k = find_entering(D, eps)

        if k == None:
            return LPResult.OPTIMAL, D

        l = find_leaving(D, k, eps)

        if l == None:
            return LPResult.UNBOUNDED, D

        D.pivot(k, l)


def run_examples():
   # Ex 1 with Fraction
    """ 
    c, A, b = example1()
    D = Dictionary(c, A, b)
    print("Example 1 with Fraction")
    # print('Initial dictionary:')
    # print(D)
    # print('x1 is entering and x4 leaving:')
    D.pivot(0, 0)
    # print(D)
    # print('x3 is entering and x6 leaving:')
    D.pivot(2, 2)
    # print(D)
    # print() """

    # Testing Blands Rule

    """ c,A,b = example1()
    D=Dictionary(c,A,b)
    print('Testing Blands Rule')
    print('Initial dictionary:')
    print(D)
    k,l = bland(D,np.finfo(np.float64).eps)
    print('Pivot :')
    if k == None: 
        print('None')
    else: 
        print(k+1)
    print('and')
    if l == None: 
        print('None')
    else: 
        print(l+1)
    D.pivot(k,l)
    print(D) """

    # Testing Largest Coefficient Rule
    """ c,A,b = example1()
    D=Dictionary(c,A,b)
    print('Testing Coefficient Rule')
    print('Initial dictionary:')
    print(D)
    k,l = largest_coefficient(D,np.finfo(np.float64).eps)
    
    print('Pivot :')
    if k == None: 
        print('None')
    else: 
        print(k+1)
    print('and')
    if l == None: 
        print('None')
    else: 
        print(l+1)
    D.pivot(k,l)
    print(D) """

    # Ex 1 with np.float64
    """
    start_time = timeit.default_timer()
    D = Dictionary(c, A, b, np.float64)
    print("Example 1 with np.float64")
    # print('Initial dictionary:')
    # print(D)
    # print('x1 is entering and x4 leaving:')
    D.pivot(0, 0)
    # print(D)
    # print('x3 is entering and x6 leaving:')
    # D.pivot(2,2)
    # print(D)
    # print()
    end_time_float = timeit.default_timer() - start_time
    """

    """ # Example 2
    c,A,b = example2()
    print('Example 2')
    print('Auxillary dictionary')
    D=Dictionary(None,A,b)
    print(D)
    print('x0 is entering and x4 leaving:')
    D.pivot(2,1)
    print(D)
    print('x2 is entering and x3 leaving:')
    D.pivot(1,0)
    print(D)
    print('x1 is entering and x0 leaving:')
    D.pivot(0,1)
    print(D)
    print()  """

    # Solve Example 1 using lp_solve
    """ c,A,b = example1()
    print('lp_solve Example 1:')
    D = Dictionary(c,A,b)
    res,D,_ = lp_solve(D)
    # print(res)
    print(D)
    print() """

    # Solve Example 2 using lp_solve
    """c, A, b = example2()
    print("lp_solve aux Example 2:")
    res, D = two_phase_solve(c, A, b)
    print(res)
    # print(D)
    print()"""

    # Solve Exercise 2.5 using lp_solve
    """c, A, b = exercise2_5()
    print("lp_solve aux Exercise 2.5:")
    res, D = two_phase_solve(c, A, b)
    print(res)
    #   print(D)
    print()"""

    # Solve Exercise 2.6 using lp_solve
    """c, A, b = exercise2_6()
    print("lp_solve Exercise 2.6:")
    res, D = two_phase_solve(c, A, b, 0)
    print(res)
    # print(D)
    print()"""

    # Solve Exercise 2.7 using lp_solve
    """
    c, A, b = exercise2_7()
    print("lp_solve Exercise 2.7:")
    res, D = two_phase_solve(c, A, b, 0)
    print(res)
    # print(D)
    print()
    """

    """ 
    #Integer pivoting
    c,A,b=example1()
    D=Dictionary(c,A,b,int)
    print('Example 1 with int')
    print('Initial dictionary:')
    print(D)
    print('x1 is entering and x4 leaving:')
    D.pivot(0,0)
    print(D)
    print('x3 is entering and x6 leaving:')
    D.pivot(2,2)
    print(D)
    print() 
    """
    """  c,A,b = integer_pivoting_example()
    D=Dictionary(c,A,b,int)
    print('Integer pivoting example from lecture')
    print('Initial dictionary:')
    print(D)
    print('x1 is entering and x3 leaving:')
    D.pivot(0,0)
    print(D)
    print('x2 is entering and x4 leaving:')
    D.pivot(1,1)
    print(D) """

    """
    
    c,A,b = cycleexample()
    D=Dictionary(c,A,b)
    print(D)
    print('lp_solve Cycle example:')
    res,D=lp_solve(c,A,b)
    print(res)
    print(D)
    print()  """

    """ c,A,b = exercise2_6()
    print('lp_solve_aux Exercise 2.6:')
    res,D=two_phase_solve(c,A,b, 0)
    print(res)
    print(D)
    print() """
    
    seed = 200000
    n = 10
    m = 10
    c, A, b = random_lp(n, m, seed)
    D = Dictionary(c, A, b, Fraction)
    print("Random with Fraction")
    start_time = timeit.default_timer()
    res1, res2 = two_phase_solve(c, A, b, Fraction)
    end_time_fraction = timeit.default_timer() - start_time
    print(res1)
    # print(res2)
    # print()

    # Random 100x100 with np.float64
    c, A, b = random_lp(n, m, seed)
    print("Random with np.float64")
    start_time = timeit.default_timer()
    res1, res2 = two_phase_solve(c, A, b, np.float64)
    end_time_float = timeit.default_timer() - start_time
    print(res1)
    # print(res2)
    # print()
    
    # Random 100x100 with SciPy linprog 
    c, A, b = random_lp(n, m, seed)
    D = Dictionary(c, A, b, np.float64)
    print("Random SciPy linprog")
    start_time = timeit.default_timer()
    res = scipy.optimize.linprog(c, A_ub=A, b_ub=b, method="highs")
    end_time_scipy = timeit.default_timer() - start_time
    print()

    print("Benchmarks:")
    print(f"Time for Fraction - Random    : {round(end_time_fraction, 7)} seconds")
    print(f"Time for np.float64 - Random  : {round(end_time_float, 7)} seconds")
    print(f"Time for SciPy - Random       : {round(end_time_scipy, 7)} seconds")
    
    xpoints = np.array([1, 8])
    ypoints = np.array([3, 10])

    # plt.plot(xpoints, ypoints, 'o')
    # plt.show()
    
run_examples()
