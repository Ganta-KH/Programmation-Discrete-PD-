import numpy as np
from sympy import Symbol, solve

def f(x1, x2, x3):
    return x1**4 + x1**2 + 2*x2**2 + x3**2 - 6*x1 + 3*x2 - 2*x3*( x1 + x2 + 1 )


def df(y ,x1, x2, x3):
    return np.array( [y.diff(x1), y.diff(x2), y.diff(x3)] )


def df2(df, x1, x2, x3):
    ddf = [ [df.item(0).diff(x1), df.item(0).diff(x2), df.item(0).diff(x3)],
            [df.item(1).diff(x1), df.item(1).diff(x2), df.item(1).diff(x3)],
            [df.item(2).diff(x1), df.item(2).diff(x2), df.item(2).diff(x3)] ]
    return np.array(ddf)


##########################################################################################

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
        print(f'iteration {i+1}: {newX} -> {nf}', )
        result[i+1] = [newX, nf]

        if stp <= epsilon: break
        x0 = newX
    return result

##########################################################################################

def Conjugate_Gradient(x, epsilon, iteration):
    result = {}
    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')

    y = f(x1, x2, x3)
    dfx = df(y, x1, x2, x3)
    k = df2(dfx, x1, x2, x3)

    for g in range(iteration):
        Q = k.copy()

        for i in range(len(Q)):
            for j in range(len(Q[i])):
                Q[i, j] = eval(str(Q[i, j]).replace("x1", str(x.item(0))).replace("x2", str(x.item(1))).replace("x3", str(x.item(2))))

        d0 = np.array([ *map(lambda dff:eval(str(dff).replace("x1", str(x.item(0))).replace("x2", str(x.item(1))).replace("x3", str(x.item(2)))), dfx), ])
        d = -d0

        alpha = -np.sum(d*d0)/np.sum(np.matmul(Q, d) * d)
        
        x11 = x + alpha*d

        fx = np.array([ *map(lambda dff: eval(str(dff).replace("x1", str(x11.item(0))).replace("x2", str(x11.item(1))).replace("x3", str(x11.item(2)))), dfx)],)

        beta = np.sum(np.matmul(Q, fx) * d) / np.sum(np.matmul(Q, d) * d)
            
        d1 = - fx + beta * d

        alpha1 = - np.sum(fx * d1) / np.sum(d1 * np.matmul(Q, d1))
        x22 = x11 + alpha1 * d1

        fx = np.array([ *map(lambda dff: eval(str(dff).replace("x1", str(x22.item(0))).replace("x2", str(x22.item(1))).replace("x3", str(x22.item(2)))), dfx)],)

        stp = (np.sqrt(np.sum(fx**2)))
        nf = f(x22.item(0), x22.item(1), x22.item(2))
        print(f'iteration {g+1}: {x22} -> {nf}')
        result[g+1] = [x22, nf]
        x = x22.copy()
        if stp <= epsilon: break
    return result

##########################################################################################

def Newton(x , epsilon, iteration):
    result = {}

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
        result[a+1] = [x, nf]
        if abs(nf - lf) < epsilon:
            break
        lf = nf
    return result