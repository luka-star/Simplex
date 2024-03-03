import numpy as np
from fractions import Fraction
from enum import Enum

def example1(): return np.array([5,4,3]),np.array([[2,3,1],[4,1,2],[3,4,2]]),np.array([5,11,8])
def example2(): return np.array([-2,-1]),np.array([[-1,1],[-1,-2],[0,1]]),np.array([-1,-2,1])
def integer_pivoting_example(): return np.array([5,2]),np.array([[3,1],[2,5]]),np.array([7,5])
def exercise2_5(): return np.array([1,3]),np.array([[-1,-1],[-1,1],[1,2]]),np.array([-3,-1,4])
def exercise2_6(): return np.array([1,3]),np.array([[-1,-1],[-1,1],[1,2]]),np.array([-3,-1,2])
def exercise2_7(): return np.array([1,3]),np.array([[-1,-1],[-1,1],[-1,2]]),np.array([-3,-1,2])
def cycleexample(): return np.array([1,-2,0,-2]),np.array([[0.5,-3.5,-2,4], 
                                                           [0.5,-1,-0.5,0.5],
                                                           [1,0,0,0]]),np.array([0,0,1])
def auxexample(): return np.array([-2,-1]),np.array([[-1,1], 
                                                     [-1,-2],
                                                     [0,1]]),np.array([-1,-2,1])

def random_lp(n,m,sigma=10): return np.round(sigma*np.random.randn(n)),np.round(sigma*np.random.randn(m,n)),np.round(sigma*np.abs(np.random.randn(m)))

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
    
    def __init__(self,c,A,b,dtype=Fraction):
        # Initializes the dictionary based on linear program in
        # standard form given by vectors and matrices 'c','A','b'.
        # Dimensions are inferred from 'A' 
        #
        # If 'c' is None it generates the auxillary dictionary for the
        # use in the standard two-phase simplex algorithm
        #
        # Every entry of the input is individually converted to the
        # given dtype.
        m,n = A.shape
        self.dtype=dtype
        if dtype == int:
            self.lastpivot=1
        if dtype in [int,Fraction]:
            dtype=object
            if c is not None:
                c=np.array(c,dtype=object)
            A=np.array(A,dtype=object)
            b=np.array(b,dtype=object)
        self.C = np.empty([m+1,n+1+(c is None)],dtype=dtype)
        self.C[0,0]=self.dtype(0)
        if c is None:
            self.C[0,1:]=self.dtype(0)
            self.C[0,n+1]=self.dtype(-1)
            self.C[1:,n+1]=self.dtype(1)
        else:
            for j in range(0,n):
                self.C[0,j+1]=self.dtype(c[j])
        for i in range(0,m):
            self.C[i+1,0]=self.dtype(b[i])
            for j in range(0,n):
                self.C[i+1,j+1]=self.dtype(-A[i,j])
                
        self.N = np.array(range(1,n+1+(c is None)))
        self.B = np.array(range(n+1+(c is None),n+1+(c is None)+m))
        self.varnames=np.empty(n+1+(c is None)+m,dtype=object)
        self.varnames[0]='z'
        for i in range(1,n+1):
            self.varnames[i]='x{}'.format(i)
        if c is None:
            self.varnames[n+1]='x0'
        for i in range(n+1,n+m+1):
            self.varnames[i+(c is None)]='x{}'.format(i)

    def __str__(self):
        # String representation of the dictionary in equation form as
        # used in Vanderbei.
        m,n = self.C.shape
        varlen = len(max(self.varnames,key=len))
        coeflen = 0
        for i in range(0,m):
            coeflen=max(coeflen,len(str(self.C[i,0])))
            for j in range(1,n):
                coeflen=max(coeflen,len(str(abs(self.C[i,j]))))
        tmp=[]
        if self.dtype==int and self.lastpivot!=1:
            tmp.append(str(self.lastpivot))
            tmp.append('*')
        tmp.append('{} = '.format(self.varnames[0]).rjust(varlen+3))
        tmp.append(str(self.C[0,0]).rjust(coeflen))
        for j in range(0,n-1):
            tmp.append(' + ' if self.C[0,j+1]>0 else ' - ')
            tmp.append(str(abs(self.C[0,j+1])).rjust(coeflen))
            tmp.append('*')
            tmp.append('{}'.format(self.varnames[self.N[j]]).rjust(varlen))
        for i in range(0,m-1):
            tmp.append('\n')
            if self.dtype==int and self.lastpivot!=1:
                tmp.append(str(self.lastpivot))
                tmp.append('*')
            tmp.append('{} = '.format(self.varnames[self.B[i]]).rjust(varlen+3))
            tmp.append(str(self.C[i+1,0]).rjust(coeflen))
            for j in range(0,n-1):
                tmp.append(' + ' if self.C[i+1,j+1]>0 else ' - ')
                tmp.append(str(abs(self.C[i+1,j+1])).rjust(coeflen))
                tmp.append('*')
                tmp.append('{}'.format(self.varnames[self.N[j]]).rjust(varlen))
        return ''.join(tmp)

    def basic_solution(self):
        # Extracts the basic solution defined by a dictionary D
        m,n = self.C.shape
        if self.dtype==int:
            x_dtype=Fraction
        else:
            x_dtype=self.dtype
        x = np.empty(n-1,x_dtype)
        x[:] = x_dtype(0)
        for i in range (0,m-1):
            if self.B[i]<n:
                if self.dtype==int:
                    x[self.B[i]-1]=Fraction(self.C[i+1,0],self.lastpivot)
                else:
                    x[self.B[i]-1]=self.C[i+1,0]
        return x

    def value(self):
        # Extracts the value of the basic solution defined by a dictionary D
        if self.dtype==int:
            return Fraction(self.C[0,0],self.lastpivot)
        else:
            return self.C[0,0]

    def pivot(self, k, l):
        # Pivot Dictionary with N[k] entering and B[l] leaving
        # Performs integer pivoting if self.dtype==int
        # l is pivot row
        # k is pivot column
        # save pivot coefficient
        print('Pivoting ',k,l)
        a = self.C[l + 1, k + 1]

        # Update all elements of the matrix except the pivot row and pivot column
        for i in range(self.C.shape[0]):
            for j in range(self.C.shape[1]):
                if i != l + 1 and j != k + 1:
                    b = self.C[i, k + 1]
                    c = self.C[l + 1, j]
                    self.C[i, j] = self.C[i, j] - (c * b) / a

        # Perform row operations for the pivot row
        self.C[l + 2:, k + 1] /= a
        self.C[: l + 1, k + 1] /= a
        self.C[l + 1, : k + 1] /= -a
        self.C[l + 1, k + 2:] /= -a
        self.C[l + 1, k + 1] = 1 / a

        # Swap entering and leaving variables
        self.N[k], self.B[l] = self.B[l], self.N[k]


class LPResult(Enum):
    OPTIMAL = 1
    INFEASIBLE = 2
    UNBOUNDED = 3

def bland(D,eps):
    # Assumes a feasible dictionary D and finds entering and leaving
    # variables according to Bland's rule.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    
    # Returns k and l such that
    # k is None if D is Optimal
    # Otherwise D.N[k] is entering variable
    # l is None if D is Unbounded
    # Otherwise D.B[l] is a leaving variable  
    k=l=None
    lowestV = 0
    for i in range(0, len(D.N)): 
        if D.C[0, i + 1] > eps:
            if D.N[i] <= D.N[lowestV]:
                lowestV = i
                k = lowestV
    
    
    # check if the dictionary is optimal
    if (k == None):
        return k, l
    
    min_ratio = (np.inf, np.inf)
    for i in range(1, D.C.shape[0]):
        if D.C[i, k] > eps or D.C[i,k] < -eps:
            ratio = np.abs(D.C[i, 0] / D.C[i, k])
            if ratio < min_ratio[0]:
                min_ratio = (ratio, i)
                l = i-1
            elif ratio == min_ratio[0]:
                if min_ratio[1] > i: 
                    l = i - 1
                    min_ratio = (ratio, i)
    

    # check if the dictionary is unbounded
    if np.all(D.C[1:,:] > eps):
      return k, None
    
    return k-1, l

 
#Find the first variable in the objective function and return its index, otherwise return None
def find_entering(D,eps):
    """ for i in range(1, len(D.N)): 
        if D.C[0, i] > eps:
            return i-1   
    return None """
    maxcoef = np.max(D.C[0,1:])  
    if maxcoef <= eps:
        return None
    else:
        return np.argmax(D.C[0, 1:]) 
        

#Choose leaving variable according to one-phase simplex
def find_leaving(D,k,eps):
    l = None
    print(D)
    min_ratio = np.inf
    min_indices = []  
    for i in range(1, D.C.shape[0]):
        if not(D.C[i, k+1] <= eps and D.C[i, k+1] >= -eps):
            ratio = np.abs(D.C[i, 0] / D.C[i, k+1]) #fjernede abs
            if min_ratio > ratio:
                min_ratio = ratio
                min_indices = [i-1]  
            elif min_ratio == ratio:
                min_indices.append(i-1)  
    if min_indices:
        l = min(min_indices)
    return l

def find_leaving_onephase(D,k,eps):
    l = None
    min_ratio = np.inf
    min_indices = []  
    for i in range(1, D.C.shape[0]):
        if not(D.C[i, k+1] <= eps and D.C[i, k+1] >= -eps):
            ratio = D.C[i, 0] / D.C[i, k+1] #fjernede abs
            if min_ratio > ratio:
                min_ratio = ratio
                min_indices = [i-1]  
            elif min_ratio == ratio:
                min_indices.append(i-1)  
    if min_indices:
        l = min(min_indices)
    return l

def largest_coefficient(D,eps):
    # Assumes a feasible dictionary D and find entering and leaving
    # variables according to the Largest Coefficient rule.
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    # Returns k and l such that
    # k is None if D is Optimal
    # Otherwise D.N[k] is entering variable
    # l is None if D is Unbounded
    # Otherwise D.B[l] is a leaving variable
    
    k=l=None
  
    # Find entering variable according to largest coefficient rule
    maxcoef = np.max(D.C[0,1:])  
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
                ratio = -(D.C[i, 0] / D.C[i, k+1])
                if ratio < min_ratio:
                    min_ratio = ratio
                    l = i-1

        
    return k, l


def largest_increase(D,eps):
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
    
    k=l=None
    # TODO
    return k,l


def phase1(D, A,b,dtype, pivotrule):
    objectiveFunction = D.C[0, :].copy() #copy the objective function
    AD = Dictionary(None,A,b,dtype) #create auxillary dictionary
    x0 = AD.N[-1].copy() #find the index of x0
    leavingVariable = np.argmin(AD.C[1:,0]) #find the index of the leaving variable
    EnteringVariable = np.shape(AD.C)[1]-2 #find the entering variable
    print("leaving", leavingVariable)
    print("entering", EnteringVariable)
    AD.pivot(EnteringVariable, leavingVariable) #do the first pivot
    print("first auxillary pivot: ")
    print(AD)
    while True: #while loop to keep pivoting until x0 is out of the basis
        if x0 in AD.N: #if x0 is a non-basic variable then the auxillary dictionary could be feasible
            if np.any(AD.C[1:,0] < 0): #if the dictionary is infeasible
                return LPResult.INFEASIBLE,None #PLACEHOLDER - x0 could return to non-basic variables multiple times in order to make dictionary feasible
            else:
                indexOfx0 = np.where(AD.N == x0) #find the index of x0
                indexOfx0 = indexOfx0[0][0]+1 #get the index of x0
                AD.N = np.delete(AD.N, indexOfx0-1) #delete x0 from the non-basic variables
                AD.C = np.delete(AD.C, indexOfx0, 1) #delete the column of x0
                for i in range(0, len(D.N)): #iterate through the non-basic variables in the original dictionary
                    if D.N[i] in AD.B: #if the non-basic variable is now a basic variable in the auxillary dictionary
                        indexOfB = np.where(AD.B == D.N[i]) #find the index of the basic variable in auxillary dictionary
                        indexOfB = indexOfB[0][0] #get the index of the basic variable
                        indexOfN = np.where(D.N == D.N[i]) #find the index of the non-basic variable in original dictionary
                        indexOfN = indexOfN[0][0] #get the index of the non-basic variable
                        coefficientOfNonBasicVar = D.C[0, indexOfN+1] #get the coefficient of the non-basic variable
                        newInstanceOfVariable = AD.C[indexOfB+1, :] * coefficientOfNonBasicVar #multiply the new expression of the basic variable by the coefficient of the non-basic variable
                        objectiveFunction = np.delete(objectiveFunction, indexOfN+1) #delete the non-basic variable that is now a basic variable in the auxillary dictionary
                        objectiveFunction = np.hstack((objectiveFunction, 0)) #add a new instance of the basic variable in the original dictionary and make it a non-basic variable
                        objectiveFunction = objectiveFunction + newInstanceOfVariable #add the new instance of the basic variable to the objective function
                        AD.C = np.delete(AD.C, 0, 0) #delete the z row or objective function-row
                        AD.C = np.vstack((objectiveFunction, AD.C)) #add the objective function back to the dictionary
                        AD #set the original dictionary to the auxillary dictionary
                        print("new original dictionary: ")
                        print(D)
                break
        k,l = pivotrule(AD) #find the entering and leaving variables
        print("k: ", k, "l: ", l)
        if k is None: #if the auxillary dictionary is optimal and x0 is still in the basis then the original dictionary is infeasible
            return LPResult.INFEASIBLE,None 
        AD.pivot(k,l) #pivot the auxillary dictionary
        print("continued auxillary pivot: ")
        print(AD)
    return AD #return the new auxillary dictionary, that serves as the original dictionary for phase 2 

#Implementing twophase Simplex using auxillary var.
def two_phase_solve(c,A,b,dtype=Fraction,eps=0):
    
    og = Dictionary(c,A,b).C[0,:]
       
    D = Dictionary(None, A,b)
    
    k = len(D.N)-1

    l = find_leaving_onephase(D,k, eps)
    
    D.pivot(k,l)
    
    res, Dres, last = lp_solve(D)
       
    if np.any(Dres.N == k+1):
        #x0 is a non-basic varible, so we can just drop it and do second phase...
        index = int(np.where(Dres.N == k+1)[0])
        
        D = drop_x0_basic(Dres,index,og, last)

        res, D, _ = lp_solve(D)
        return res, D
    else:
        #x0 is a basic variable, so we have to do something different...
        index = int(np.where(Dres.B == k+1)[0])
        entering = -1 
        for i in range(len(D.N)):
            if D.N[i] <= len(c):
                entering = i
                break 
        D.pivot(entering, index)
        
        D = drop_x0_basic(Dres,index,og, last)
        
        res, D, _ = lp_solve(D)

        return res, D

            
    
def drop_x0_nonbasic(D,i,c):
    # Drop the artificial variable x0 from the basic variables list
    print("c",c)

    numberN0 = len(D.N)
    basic = D.B 
    nonbasic = np.delete(D.N,i) #delete index for x0
    
    for j in range(len(basic)):
        if numberN0 < basic[j]:
            basic[j] -=1

    for q in range(len(nonbasic)):
        if numberN0 < nonbasic[q]:
            nonbasic[q] -=1
                
    newC = np.delete(D.C, i + 1, axis=1) 
    
    D=Dictionary(newC[0,1:],-1*newC[1:,1:],newC[1:,0]) # construct new dictionary  
    D.C[0] = newC[0]
    D.N = nonbasic
    D.B = basic
 
    return D
            
def drop_x0_basic(D,i,c, last):
    # Drop the artificial variable x0 from the basic variables list
    numberN0 = len(D.N)
    basic = D.B 
    nonbasic = np.delete(D.N,i) #delete index for x0
    
    for j in range(len(basic)):
        if numberN0 < basic[j]:
            basic[j] -=1

    for q in range(len(nonbasic)):
        if numberN0 < nonbasic[q]:
            nonbasic[q] -=1
            
    c = np.append(c, 0)
    
    D.C[0] = c - last
    newC = np.delete(D.C, i + 1, axis=1) 
    
    D=Dictionary(newC[0,1:],-1*newC[1:,1:],newC[1:,0]) # construct new dictionary  
    D.C[0]=newC[0]
    
    D.N = nonbasic
    D.B = basic
    
    for i in range(len(D.C[0, :])): 
        D.C[0, i] -= last[i] 
 
    return D
    
 
 
def lp_solve(D,eps=0,pivotrule=lambda D: bland(D,eps=0),verbose=False):
    last_obj_func = None
    while(True):
        if(not(np.all(D.C[1:,0] >= eps)) and np.all(D.C[0, 1:] < -eps)):
            return LPResult.INFEASIBLE, None, last_obj_func
    
        k = find_entering(D,eps)
        
        if(k == None):
            return LPResult.OPTIMAL, D, last_obj_func
        
        l = find_leaving(D,k,eps)        
        
        if(l == None):
            return LPResult.UNBOUNDED, None, last_obj_func
        
        last_obj_func = (D.C[0, :].copy())
      
        D.pivot(k,l) 
  

  
def run_examples():
    # Example 1
    """ c,A,b = example1()
    D=Dictionary(c,A,b)
    print('Example 1 with Fraction')
    print('Initial dictionary:')
    print(D)
    print('x1 is entering and x4 leaving:')
    D.pivot(0,0)
    print(D)
    print('x3 is entering and x6 leaving:')
    D.pivot(2,2)
    print(D)
    print() """ 
    

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
    

    """   D=Dictionary(c,A,b,np.float64)
    print('Example 1 with np.float64')
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
    print(res)
    print(D)
    print() """

    # Solve Example 2 using lp_solve
    c,A,b = example2()
    print('lp_solve aux Example 2:')
    res,D=two_phase_solve(c,A,b)
    print(res)
    print(D)
    print() 
    

    # Solve Exercise 2.5 using lp_solve
    """ c,A,b = exercise2_5()
    print('lp_solve aux Exercise 2.5:')
    res,D=two_phase_solve(c,A,b)
    print(res)
    print(D)
    print() """  

    # Solve Exercise 2.6 using lp_solve
    """ c,A,b = exercise2_6()
    print('lp_solve Exercise 2.6:')
    res,D=two_phase_solve(c,A,b, 0)
    print(res)
    print(D)
    print() """

    """ # Solve Exercise 2.7 using lp_solve
    c,A,b = exercise2_7()
    print('lp_solve Exercise 2.7:')
    res,D=lp_solve(c,A,b)
    print(res)
    print(D)
    print()  """
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

    
    

    
run_examples()

