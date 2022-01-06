from sympy import Symbol
import numpy as np
import time
from tools import f, df, df2

def Conjugate_Gradient(x, epsilon, iteration):

    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')

    y = f(x1, x2, x3)
    print("y = ",y)
    dfx = df(y, x1, x2, x3)

    k = df2(dfx, x1, x2, x3)

    for g in range(iteration):
        Q = k.copy()

        for i in range(len(Q)):
            for j in range(len(Q[i])):
                Q[i, j] = eval(str(Q[i, j]).replace("x1", str(x.item(0))).replace("x2", str(x.item(1))).replace("x3", str(x.item(2))))

        d0 = np.array([ *map(lambda dff:eval(str(dff).replace("x1", str(x.item(0))).replace("x2", str(x.item(1))).replace("x3", str(x.item(2)))), dfx), ])
        d = -d0
        print("d0: ",d)

        alpha = -np.sum(d*d0)/np.sum(np.matmul(Q, d) * d)
        
        x11 = x + alpha*d

        fx = np.array([ *map(lambda dff: eval(str(dff).replace("x1", str(x11.item(0))).replace("x2", str(x11.item(1))).replace("x3", str(x11.item(2)))), dfx)],)
 
        beta = np.sum(np.matmul(Q, fx) * d) / np.sum(np.matmul(Q, d) * d)
            
        d1 = - fx + beta * d

        alpha1 = - np.sum(fx * d1) / np.sum([d1 * np.matmul(Q, d1)])
        x22 = x11 + alpha1 * d1

        fx = np.array([ *map(lambda dff: eval(str(dff).replace("x1", str(x22.item(0))).replace("x2", str(x22.item(1))).replace("x3", str(x22.item(2)))), dfx)],)

        stp = (np.sqrt(np.sum(fx**2)))
        nf = f(x22.item(0), x22.item(1), x22.item(2))
        print(f'iteration {g+1}: {x22} -> {nf}')
        x = x22.copy()
        if stp <= epsilon: break
        
    return x

start_time = time.time()

Conjugate_Gradient(np.array([1.0, 1.0, 1.0]), 1e-2, 150)

print("--- %s seconds ---" % (time.time() - start_time))
