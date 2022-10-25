import numpy as np
# Inputs:
# Stepsize:                 n (\eta)
# Stochastic Gradient:      g
# Start:                    x
# Stopping criterium:       sc


def gradient_descend(g, n, x, sc = 10e-8):
    it = [x]
    cur = x
    while np.linalg.norm(g(cur)) >= sc:
        gr = g(cur)
        cur = cur - n * gr/np.linalg.norm(gr)
        it.append(cur)
    return it

# Inputs
# Epsilon:                              e (\epsilon)

def adagrad(g, n, x, sc = 10e-8, e = 10e-8):
    it = [x]
    u = np.zeros(np.shape(x))                           
    cur = x
    while np.linalg.norm(g(cur)) >= sc:
        gr = g(cur)
        u = u + gr**2
        cur = cur-n*gr/(u**0.5+e)
        it.append(cur)
    return it

def rmsprop(g, n, x, sc = 10e-8):
    pass

# Inputs:
# First-order momentum parameter:       a (\alpha) 
# Second-order momentum parameter:      b (\beta)


def adam(g, n, x, sc = 10e-8, a = 0.9, b = 0.999, e = 10e-8):
    it = [x]
    v = np.zeros(np.shape(x))                           #v_t (first-order gradient with momentum)
    u = np.zeros(np.shape(x))                           #u_t (second-order gradient with momentum)
    cur = x
    while np.linalg.norm(g(cur)) >= sc:
        gr = g(cur)
        v = a*v+(1-a)*gr
        u = b*u+(1-b)*(gr**2)
        v_til = v/(1-a)
        u_til = u/(1-b) 
        cur = cur-n*(v_til)/(u_til**0.5+e)
        #print(np.array2string(gr)+" "+np.array2string(v)+" "+np.array2string(u)+" "+np.array2string(cur)+" ")
        it.append(cur)
    return it
