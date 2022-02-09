import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import spock
from spock import FeatureClassifier
import corner
from tqdm import tqdm
from scipy import stats
from celmech import Andoyer
from statsmodels.stats.weightstats import DescrStatsW
import warnings
import rebound
import timeit
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

t_orb = t_orbits*5.98
tmax = 1.0*t_orb
Nout = 200

t = np.linspace(0,tmax,Nout)
obs_tess = np.where(np.round(abs(t - t_orb),6) == np.round(np.min(abs(t - t_orb)),6))[0][0]
delta_index = 0
obs_idx=[delta_index,obs_tess+delta_index]

nobs = 2
obs_err = np.array([0.0001, 0.0005])
obs = np.array([1.503, 1.503])

# last TESS observation:
t_tess = 4664.65

# last K2 observation:
t_K2 = 2265

t = t_tess - t_K2

t_orbits = t/8.24958

t_orb = t_orbits*5.98
tmax = 1.0*t_orb
Nout = 200

t = np.linspace(0,tmax,Nout)
obs_tess = np.where(np.round(abs(t - t_orb),6) == np.round(np.min(abs(t - t_orb)),6))[0][0]
delta_index = 0
obs_idx=[delta_index,obs_tess+delta_index]

nobs = 2
obs_err = np.array([0.0001, 0.0005])
obs = np.array([1.503, 1.503])

def makesim(theta):


    e_forced, e_free, mu, deltaT, mb, eb, pomegab, thetab = theta

    mratio=0.5
    e_com = 0.0
    phiecom=float(np.random.uniform(0, 2*np.pi, size=1)) # varying between 0 and 2pi
    theta1 = np.pi
    Mstar = 1.1
    m1 = mratio*10**mu
    m2 = (1-mratio)*10**mu
    phi = np.pi # where equilibrium is
    theta1 = np.pi # position of planet b

    andvars = Andoyer.from_Z(j=3, k=1, Z=(e_forced+e_free)/np.sqrt(2), phi=phi,
                             Zstar=e_forced/np.sqrt(2), Mstar=Mstar, m1=m1, m2=m2,
                             Zcom=e_com/np.sqrt(2), phiZcom=phiecom, theta1=theta1)

    try:

        sim = andvars.to_Simulation()
        sim.add(m=mb, P=sim.particles[2].P*1.946, e=eb, pomega=pomegab, theta=thetab)
        sim.integrator="whfast"
        sim.dt = sim.particles[1].P/10
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

    if theta.ndim > 1:
        sims = [makesim(param) for param in theta]

        period_ratios = [run(sim) for sim in sims]



        return [periods[obs_idx] for periods in period_ratios]

    else:
        sim = makesim(theta)
        period_ratios = run(sim)
        return period_ratios[obs_idx]

def gen_priors_array(seed, n):

    size_arr = int(n)

    rng = np.random.default_rng(seed)

    eforced_0 = rng.uniform(0.0005,0.23,size=size_arr) # changing the upper limit to the crossing eccentricity
    efree_0 = rng.uniform(0.0005,0.003,size=size_arr) # changing the bounds because experiments show that
    mu_0 = rng.uniform(np.log10(8.964e-6), np.log10(5.25e-4), size=size_arr) # new upper limit of 0.55 Mjup by Mascareno

    mb_0 = stats.norm.rvs(loc=0.69, scale=0.19, size=size_arr)

    eb_0 = stats.norm.rvs(loc=0.13, scale=0.07, size=size_arr)

    pomegab_0 = np.random.uniform(0, 2*np.pi, size=size_arr)

    thetab_0 = np.random.uniform(0, 2*np.pi, size=size_arr)

    deltaT_0 = rng.uniform(0, 2000, size=size_arr)

    prior_set = np.column_stack((eforced_0, efree_0, mu_0, deltaT_0, mb_0, eb_0, pomegab_0, thetab_0))

    valid_priors = []
    for param in prior_set:
        if param[4] < 0 or param[5] < 0:
            continue
        else:
            valid_priors.append(param)

    return valid_priors

def lnlike(theta):

    #e_forced, e_free, deltaT, mu, mb, eb, pomegab, thetab = theta

    model = compute_model(theta)
    argument = (obs - model)**2 / obs_err**2
    if theta.ndim > 1:
        loglike = [0.5*np.sum(arg) for arg in argument]
    else:
        loglike = 0.5*np.sum(argument)
    return loglike

def get_posteriors(prior):

    like_val = lnlike(prior)

    rng = np.random.default_rng(0)

    prob = np.log10(rng.random())

    if -like_val > prob:
        post = prior

        return post
    else:
    	return np.nan


priors = gen_priors_array(0, sys.argv[1])
#seeds = list(range(int(sys.argv[1]),int(sys.argv[2])))


#seed_batches = [seeds[i:i + n] for i in range(0, len(seeds), n)]
start_time = timeit.default_timer()
#for i,seed_batch in enumerate(seed_batches):
if __name__ == '__main__':
    #pool = Pool()
    #results = pool.map(get_posteriors,priors)
    results = p_map(get_posteriors,priors)
    #pool.close()
    #pool.join()
    #np.save('/Users/Helios/gdrive_pu/tamayo_research/lnlike_100mil/batch_{}.npy'.format(i+576), results) # 25 mil broke after file 257 so replacing format so it doesn't overwrite
    print("--- %s seconds ---" % (timeit.default_timer() - start_time))

    print(len(np.array(results)[~np.isnan(results)]))
