import numpy as np
from scipy.optimize import root_scalar
from numpy.linalg import vector_norm
from scipy import linalg
from scipy.integrate import solve_ivp

def jacobi(data,mu):
    # Data (time,state)
    r1 = np.sqrt((data[:,0]+mu)**2+(data[:,1])**2+(data[:,2])**2)
    r2 = np.sqrt((data[:,0]+mu-1)**2+(data[:,1])**2+(data[:,2])**2)
    return (vector_norm(data[:,0:2],axis=1)**2)+2*(1-mu)/r1+2*mu/r2-vector_norm(data[:,3:],axis=1)**2

def jacobiBatch(data,mu):
    # Data (orbit,time,state)
    r1 = np.sqrt((data[:,:,0]+mu)**2+(data[:,:,1])**2+(data[:,:,2])**2)
    r2 = np.sqrt((data[:,:,0]+mu-1)**2+(data[:,:,1])**2+(data[:,:,2])**2)
    return (vector_norm(data[:,:,0:2],axis=2)**2)+2*(1-mu)/r1+2*mu/r2-vector_norm(data[:,:,3:],axis=2)**2


def coLinearLagrangePts(mu):
    """
    Calculates the locations of the collinear Lagrange points (L1, L2, L3)
    in the Circular Restricted Three-Body Problem (CR3BP).

    Inputs: mu (double)

    Outputs: l1, l2, l3 x-coordinates (doubles)
    """

    def func(x):
        return x - (1 - mu) * (x + mu) / (abs(x + mu)**3) - mu * (x - 1 + mu) / (abs(x - 1 + mu)**3)

    l1 = root_scalar(func, bracket=[0.8, 0.9]).root
    l2 = root_scalar(func, bracket=[1.1, 1.2]).root
    l3 = root_scalar(func, bracket=[-1.1, -0.9]).root

    return l1, l2, l3

def gradU(state, mu):
  """Calculates the gradient of the potential function U.
    Inputs: state (numpy array)
            mu (double) 
    Outputs: gradient 1x3 double array
  """
  r1 = np.sqrt((state[0] + mu)**2 + state[1]**2 + state[2]**2)
  r2 = np.sqrt((state[0] - 1 + mu)**2 + state[1]**2 + state[2]**2)
  grad = [
      state[0] - (1 - mu) * (state[0] + mu) / r1**3 - mu * (state[0] - 1 + mu) / r2**3,
      state[1] - (1 - mu) * state[1] / r1**3 - mu * state[1] / r2**3,
      -(1 - mu) * state[2] / r1**3 - mu * state[2] / r2**3,
  ]
  return grad

def cr3bp(t,x,mu):
    xPOS = x[0:6]
    grad = gradU(xPOS, mu)
    xPOS_dot = [xPOS[3], xPOS[4], xPOS[5], 2*xPOS[4]+grad[0], -2*xPOS[3]+grad[1], grad[2]]

    xSTM = np.reshape(x[6:], (6, 6))
    dSTM = stmJacobian(x, mu)
    xSTM_dot = np.reshape(np.matmul(dSTM,xSTM), 36)

    return np.concatenate((xPOS_dot, xSTM_dot))

def stmJacobian(x, mu):
    s = np.zeros((6,6))
    s[0:3,3:6] = np.eye(3)
    s[3:6,0:3] = uJacobian(x, mu)
    s[3,4] = 2
    s[4,3] = -2
    return s

def uJacobian(state,mu):
    x = state[0]
    y = state[1]
    z = state[2]
    r1 = np.sqrt((x+mu)**2+y**2+z**2)
    r2 = np.sqrt((x+mu-1)**2+y**2+z**2)
    uxx = 1-(1-mu)/r1**3-mu/r2**3+3*(1-mu)*(x+mu)**2/r1**5+3*mu*(x-1+mu)**2/r2**5
    uxy = 3*(1-mu)*(x+mu)*y/r1**5+3*mu*(x-1+mu)*y/r2**5
    uxz = 3*(1-mu)*(x+mu)*z/r1**5+3*mu*(x-1+mu)*z/r2**5
    uyy = 1-(1-mu)/r1**3-mu/r2**3+3*(1-mu)*y**2/r1**5+3*mu*y**2./r2**5
    uyz = 3*(1-mu)*y*z/r1**5+3*mu*y*z/r2**5
    uzz = -(1-mu)/r1**3-mu/r2**3+3*(1-mu)*z**2/r1**5+3*mu*z**2/r2**5

    return [[uxx,uxy,uxz],
            [uxy,uyy, uyz],
            [uxz, uyz,uzz]]

def cr3bpStateOnly(t, state, mu):
  """Calculates the state derivative for the CR3BP."""
  grad = gradU(state, mu)
  deriv = [
      state[3],
      state[4],
      state[5],
      2 * state[4] + grad[0],
      -2 * state[3] + grad[1],
      grad[2],
  ]
  return np.array(deriv)

def correct_orbit_full(i, data, mu):
    v =np.zeros(7)
    v[0:6] = data[i,0:6]
    v[6] = data[i,7]
    iter = 1
    errors = []

    while iter<=100:
        init  = np.concatenate((v[0:6],np.reshape(np.eye(6),36)))
        sol = solve_ivp(
            lambda t, state: cr3bp(t, state, mu),(0, v[6]),init,method='DOP853', rtol=1e-12, atol=1e-14)

        t = sol.t
        y = sol.y.T
        x_f = y[-1,0:6]
        STM_f = np.reshape(y[-1,6:], (6, 6))

        DF = np.zeros((6,7))
        DF[0:6, 0:6] = STM_f-np.eye(6)
        DF[0:6, 6] = cr3bpStateOnly(0,x_f,mu)
        errors.append(np.linalg.norm(x_f-v[0:6]))

        if np.linalg.norm(x_f-v[0:6])<1e-10:
            break
        else:
            v = v - np.matmul(np.matmul(DF.T,np.linalg.pinv(np.matmul(DF,DF.T))),(x_f - v[0:6]))
        
        iter=iter+1

        if iter > 100:
            print("Failed to Converge")

    return [y[:,0:6],t]

def multipleShooting(x_init, mu):
    """Implement the multiple shooting method for CR3BP periodic orbits."""
    max_iter = 12
    tol = 1e-9
    l1 = [0.836915125772393,0,0]
    dt = np.sqrt((x_init[:, 0] - l1[0])**2 + x_init[:, 1]**2) * 2 * np.pi / np.shape(x_init)[0] / np.sqrt(x_init[:, 2]**2 + x_init[:, 3]**2)
    v = np.zeros((x_init.shape[0], 7))
    v[:, 0:6] = x_init
    v[:, 6] = dt

    iters = 1
    errors = np.zeros(max_iter)

    while iters <= max_iter:

        F = np.zeros(6 * x_init.shape[0])
        DF = np.zeros((6 * x_init.shape[0], 7 * x_init.shape[0]))
        for ii in range(x_init.shape[0]):
            def ode_func(t, y):
                return cr3bp(t, y, mu)
            y0 = np.concatenate([v[ii, 0:6], np.eye(6).flatten()])
            sol = solve_ivp(ode_func, [0, v[ii, 6]], y0, method='DOP853', rtol=1e-12, atol=1e-14)
            y_end = sol.y[:, -1]
            F[ii * 6:(ii + 1) * 6] = y_end[0:6]
            STM_f = y_end[6:].reshape((6, 6))
            DF[ii * 6:(ii + 1) * 6, ii * 7:ii * 7 + 6] = STM_f
            DF[ii * 6:(ii + 1) * 6, ii * 7 + 6] = cr3bpStateOnly(0, y_end[0:6], mu)

        v_stacked = np.roll(v[:, 0:6], -1, axis=0).flatten()
        F = F - v_stacked

        n = x_init.shape[0]
        for ii in range(n - 1):
            DF[ii * 6:(ii + 1) * 6, (ii + 1) * 7:(ii + 1) * 7 + 6] -= np.eye(6)
        DF[(n - 1) * 6:n * 6, 0:6] -= np.eye(6)

        if iters == 1:
            print(F)
            print(DF)

        errors[iters - 1] = np.linalg.norm(F)
        print(errors[iters - 1])
        if np.linalg.norm(F) < tol:
            break
        else:
            v_flat = v.flatten()
            delta_v = DF.T @ linalg.lstsq(DF @ DF.T, F)[0]
            v_flat = v_flat - delta_v
            v = v_flat.reshape((n,7))
        iters += 1

    if iters > max_iter:
        print("Failed to Converge")

    return v, iters, errors



