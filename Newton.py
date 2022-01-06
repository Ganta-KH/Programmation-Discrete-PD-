import numpy as np
from sympy import Symbol
import time
from tools import f, df, df2

def Newton(x , epsilon, iteration):

    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')

    y = f(x1, x2, x3)
    dx = df(y, x1, x2, x3)
    k = df2(dx, x1, x2, x3)

    lf = float("inf")

    for a in range(iteration):
        H = k.copy()

        for i in range(len(H)):
            for j in range(len(H[i])):
                H[i, j] = eval(str(H[i, j]).replace("x1", str(x.item(0))).replace("x2", str(x.item(1))).replace("x3", str(x.item(2))))

        H = np.linalg.inv(H.astype("float"))

        x1 = np.matmul(H, dx)
        x = x - np.array([*map(lambda dff:eval(str(dff).replace("x1", str(x.item(0))).replace("x2", str(x.item(1))).replace("x3", str(x.item(2)))), x1),])
        nf = f(x.item(0), x.item(1), x.item(2))
        print(f'iteration {a+1}: {x} -> {nf}')
        if abs(nf - lf) < epsilon:
            break
        lf = nf


start_time = time.time()

Newton(np.random.rand(3) * 255, 0.1, 150)

print("--- %s seconds ---" % (time.time() - start_time))

