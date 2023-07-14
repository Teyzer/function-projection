from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp

import math

import scipy.integrate as integrate

import inspect


DEFAULT_DOMAIN = (-10, 10)


class Function:
    
    def __init__(self, _func, _domain = DEFAULT_DOMAIN, _definition = None) -> None:
        
        self.func = _func
        self.domain = _domain
        
        if _definition is not None:
            self.definition = _definition
        else:
            cur = inspect.getsource(self.func).replace(": ", ":").split(":")[1].split(",")[0]
            if cur.count("(") < cur.count(")"):
                self.definition = cur[:-2]
            else:
                self.definition = cur
        
    
    @staticmethod
    def common_definition(f1, f2, sign) -> str:
        return "(" + f1.definition + ")" + sign + "(" + f2.definition + ")"
        
    @staticmethod
    def common_domain(f1, f2) -> tuple:
        mi = max(f1.domain[0], f2.domain[0])
        ma = min(f1.domain[1], f2.domain[1])
        return (mi, ma)
    
    @staticmethod
    def common_function(f1, f2, operator, merging_function):
        
        if isinstance(f2, Function):
            return Function(
                lambda *args: merging_function(f1.func(*args), f2.func(*args)),
                Function.common_domain(f1, f2),
                Function.common_definition(f1, f2, operator)
            )
        elif isinstance(f2, float) or isinstance(f2, int):
            return Function(
                lambda *args: merging_function(f1.func(*args), f2),
                f1.domain,
                "(" + f1.definition + ")" + operator + str(round(f2, 6))
            )
        
    @staticmethod
    def dot_product(f1, f2):
        g = f1 * f2
        a, b = g.domain
        i, e = integrate.quad(g, a, b)
        return i
    
        
    def __call__(self, *args) -> float:
        if isinstance(args, list):
            return [self.func(*x) for x in args]
        return self.func(*args)

    def __add__(self, other):
        return Function.common_function(self, other, "+", lambda x, y: x + y)

    def __sub__(self, other):
        return Function.common_function(self, other, "-", lambda x, y: x - y)

    def __mul__(self, other):
        return Function.common_function(self, other, "*", lambda x, y: x * y)

    def __truediv__(self, other):
        return Function.common_function(self, other, "/", lambda x, y: x / y)

    def __pow__(self, other):
        return Function.common_function(self, other, "**", lambda x, y: x ** y)
    
    def norm(self):
        return math.sqrt(Function.dot_product(self, self))
    
    def __str__(self):
        return self.definition.replace("**", "^")
    
    def __repr__(self):
        return self.__str__()
    
    def plot(self):
        plt.close()
        X = np.linspace(self.domain[0], self.domain[1])
        Y = self.__call__(X)
        plt.plot(X, Y)
        plt.grid()
        plt.show()
        plt.close()



class Space:
    
    def __init__(self, span):
        
        ortho_basis = []
        self.dim = len(span)
        
        if self.dim == 0:
            self.ob = []
            return
        
        for i, f in enumerate(span):
            print(i)
            cur = f
            for g in ortho_basis:
                cur = cur - g * (Function.dot_product(g, f) / Function.dot_product(g, g))
            ortho_basis.append(cur)
            
        orthonormal = [f / f.norm() for f in ortho_basis]
        self.base = orthonormal
        
    def project_coefs(self, f: Function):
        return [Function.dot_product(g, f) for g in self.base]
    
    def project(self, f: Function):
        coefs = self.project_coefs(f)
        res = Function(lambda x: 0)
        for c, j in zip(coefs, self.base):
            res = res + j * c
        return res

if __name__ == "__main__":

    d = 2
    
    b0 = Function(lambda x: 1)
    b1 = Function(lambda x: x)
    b2 = Function(lambda x: x**2)
    b3 = Function(lambda x: x**3)
    b4 = Function(lambda x: x**4)
    b5 = Function(lambda x: x**5)
    b6 = Function(lambda x: x**6)
    b7 = Function(lambda x: x**7)
    b8 = Function(lambda x: x**8)
    b9 = Function(lambda x: x**9)
    b10 = Function(lambda x: x**10)
    b11 = Function(lambda x: x**11)
    b12 = Function(lambda x: x**12)
    b13 = Function(lambda x: x**13)
    b14 = Function(lambda x: x**14)
    b15 = Function(lambda x: x**15)
    b16 = Function(lambda x: x**16)
    b17 = Function(lambda x: x**17)
    b18 = Function(lambda x: x**18)
    b19 = Function(lambda x: x**19)
    b20 = Function(lambda x: x**20)
    b21 = Function(lambda x: x**21)
    
    # print()
    
    b = [b0, b1, b2, b3, b4, b5, b6, 
         b7, b8, b9, b10, b11, b12, b13, b14,
         b15, b16, b17, b18]
    
    s = Space(b)
    # for pol in s.base:
    #     print("\n", pol)
    
    g = Function(lambda x: np.tanh(x))
    
    print(s.project_coefs(g))
    
    t = s.project(g)
    
    X = np.linspace(-10, 10)
    z = np.polyfit(X, t(X), 11)
    
    t.plot()
    
    print(list(z))
    
    # for i, c in enumerate(list(z)):
    #     print(str(c).replace("e-", "10^{")+"}" + "x^{" + str(i) + "} + ", end=" ")
    # print()
    
    # print(t)
    