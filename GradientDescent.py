from sympy import Symbol, solve
import numpy as np
import time
from tools import f, df

def GradientDescentD3(x0, epsilon, iteration):
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    alpha = Symbol('a')

    y = f(x1, x2, x3)
    dx = df(y, x1, x2, x3)

    result = {}

    for i in range(iteration):

        dfx = np.array([ *map(lambda dff: eval(str(dff).replace("x1", str(x0.item(0))).replace("x2", str(x0.item(1))).replace("x3", str(x0.item(2)))), dx)],)
 
        grd1 = x0.item(0) - alpha*dfx.item(0)
        grd2 = x0.item(1) - alpha*dfx.item(1)
        grd3 = x0.item(2) - alpha*dfx.item(2)

        fa = f(grd1, grd2, grd3)

        da = fa.diff(alpha)
        alphaa = solve(da)

        alphaa = alphaa[0]
        newX = np.array([x0.item(0) - alphaa*dfx.item(0), x0.item(1) - alphaa*dfx.item(1), x0.item(2) - alphaa*dfx.item(2)])

        stp = np.sqrt(np.sum( np.array([ *map(lambda dff: eval(str(dff).replace("x1", str(newX.item(0))).replace("x2", str(newX.item(1))).replace("x3", str(newX.item(2)))), dx)],)**2 ))

        nf = f(newX.item(0), newX.item(1), newX.item(2))
        print("result: ",newX)
        print(f'iteration {i+1}: {nf}', )
        result[i+1] = [newX, nf]

        if stp <= epsilon: break
        x0 = newX
    return result

start_time = time.time()

#r = GradientDescentD3(np.random.rand(3), 0.1, 150)
#
#print("--- %s seconds ---" % (time.time() - start_time))
