import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import spock
from spock import FeatureClassifier
import corner
from tqdm import tqdm
from scipy import stats
#from scipy.optimize import minimize
from celmech import Andoyer
#from celmech.disturbing_function import get_fg_coeffs
from statsmodels.stats.weightstats import DescrStatsW
import warnings
import rebound
import time
import sys
from multiprocessing import Pool
from p_tqdm import p_map
warnings.simplefilter(action='ignore', category=FutureWarning)


# last TESS observation:
t_tess = 4664.65

# last K2 observation:
t_K2 = 2265

t = t_tess - t_K2

t_orbits = t/8.24958

#print('Number of orbits between observations: {:.0f}'.format(t_orbits))

t_orb = t_orbits*5.98
tmax = 1.0*t_orb
Nout = 200

t = np.linspace(0,tmax,Nout)
obs_tess = np.where(np.round(abs(t - t_orb),6) == np.round(np.min(abs(t - t_orb)),6))[0][0] 

delta_index = 0
obs_idx=[delta_index,obs_tess+delta_index] # [20, 519] in the current version

nobs = 2
obs_err = np.array([0.0001, 0.0005])
obs = np.array([1.503, 1.503]) 

def makesim(theta):
    
    e_forced, e_free, deltaT, mu = theta

    mratio=0.5
    #e_com=float(np.random.uniform(0, 0.3, size=1)),  # varying between 0 and max(Z*) = 0.3
    e_com = 0.0
    phiecom=float(np.random.uniform(0, 2*np.pi, size=1)) # varying between 0 and 2pi
    #phiecom=0.0
    theta1 = np.pi
    #pomega_b=None
    
    Mstar = 1.1
    m1 = mratio*10**mu
    m2 = (1-mratio)*10**mu
    phi = np.pi # where equilibrium is
    theta1 = np.pi # so they start on opposite sides

    andvars = Andoyer.from_Z(j=3, k=1, Z=(e_forced+e_free)/np.sqrt(2), phi=phi, 
                             Zstar=e_forced/np.sqrt(2), Mstar=Mstar, m1=m1, m2=m2, 
                             Zcom=e_com/np.sqrt(2), phiZcom=phiecom, theta1=theta1)

    try:
        
        sim = andvars.to_Simulation()
        sim.integrator="whfast"
        sim.dt = sim.particles[1].P/20
        sim.ri_whfast.safe_mode = 0
        sim.integrate(deltaT)
        sim.t = 0
        return sim
    except:
        print(e_forced, e_free, deltaT, mu)
        raise
    
def run(sim):
    
    Pratios = np.zeros(Nout)
    ps = sim.particles

    for i, time in enumerate(t):
        
        sim.integrate(time)
        Pratios[i] = ps[2].P/ps[1].P
        
    return Pratios

def compute_model(theta):
    """
    Computes the model observation periods given a set of parameters at the observation indices
    """

    sim = makesim(theta)
    
    period_ratios = run(sim)

    return period_ratios[obs_idx]

def gen_priors(seed):

    rng = np.random.default_rng(seed)

    eforced_0 = float(rng.uniform(0.0005,0.2,size=1))
    efree_0 = float(rng.uniform(0.0005,0.2,size=1))
    mu_0 = float(rng.uniform(np.log10(3.0027e-5), -3, size=1)) # 01/20/22: Testing a higher upper limit by a factor of 3 (changed from 3 Mearth to 10 Mearth)
    delta_T0 = float(rng.uniform(0, 2000, size=1))

    par = (eforced_0,efree_0,delta_T0, mu_0)
    
    return par
    
def lnlike(theta):
    
    """
    Determines the gaussian log likelihood.
    obs: period ratio observations
    theta: parameters
    obs_err: errors in the shape of (obs,sample_size) for K2 and TESS
    """
    e_forced, e_free, deltaT, mu = theta
    
    model = compute_model(theta)
    argument = (obs - model)**2 / obs_err**2

    loglike = 0.5*np.sum(argument)
    return loglike

def get_posteriors(seed):
    
    param_prior = gen_priors(seed)
    #param_prior = (0.0943571261630648, 0.004654693027856666, 1442.7862329157383, -3.8074133185131265) # this is a test case 
    like_val = -lnlike(param_prior)

    rng = np.random.default_rng(seed)
    
    prob = np.log10(rng.random())
    
    if like_val > prob:
        post = param_prior

        return post

    else: 
    	return np.nan
        #like_val.append(val)
    
    

#seeds = list(range(0,int(sys.argv[1])))

#seeds = list(range(5000000,9000000))

seeds = list(range(int(sys.argv[1]),int(sys.argv[2])))

n = int(1e5)

seed_batches = [seeds[i:i + n] for i in range(0, len(seeds), n)]

#start_time = time.time()

for i,seed_batch in enumerate(seed_batches):
	if __name__ == '__main__':
		#pool = Pool()
		#results = pool.map(get_posteriors,seeds)
		results = p_map(get_posteriors,seed_batch)
		#pool.close()
		#pool.join()
		np.save('lnlike_test_posteriors/batches_10Mearth_1Mjup_c_d_30mil/batch_{}.npy'.format(i+534), results) # 25 mil broke after file 257 so replacing format so it doesn't overwrite
		#print("--- %s seconds ---" % (time.time() - start_time))




