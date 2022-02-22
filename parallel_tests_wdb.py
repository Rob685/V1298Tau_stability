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
obs_err_db = np.array([0.0001, 0.0004])
obs = np.array([1.503, 1.503])
obs_db = np.array([1.946, 1.946])

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
        sim.dt = sim.particles[1].P/20
        sim.ri_whfast.safe_mode = 0
        sim.integrate(deltaT)
        sim.t = 0
        return sim
    except KeyboardInterrupt:
        print('Interrupted')
        print(e_forced, e_free, mu, deltaT, mb, eb, pomegab, thetab)
        raise

def run(sim):

    Pratios_cd = np.zeros(Nout)
    Pratios_db = np.zeros(Nout)
    ps = sim.particles

    for i, time in enumerate(t):

        sim.integrate(time)
        Pratios_cd[i] = ps[2].P/ps[1].P
        Pratios_db[i] = ps[3].P/ps[2].P

    return Pratios_cd, Pratios_db

def gen_priors_array(seed):

    #size_arr = int(n)

    rng = np.random.default_rng(seed)

    eforced_0 = float(rng.uniform(0.0005,0.23,size=1)) # changing the upper limit to the crossing eccentricity
    efree_0 = float(rng.uniform(0.0005,0.04,size=1))
    mu_0 = float(rng.uniform(np.log10(8.964e-6), np.log10(5.25e-4), size=1)) # new upper limit of 0.55 Mjup by Mascareno

    #mb_0 = stats.norm.rvs(loc=0.69, scale=0.19, size=size_arr)*9.54e-4
    mb_0 = float(stats.truncnorm.rvs(a=(0 - 0.69)/0.19, b = np.inf, loc=0.69, scale=0.19, size=1))*9.54e-4
    #eb_0 = stats.norm.rvs(loc=0.13, scale=0.07, size=size_arr)
    eb_0 = float(stats.truncnorm.rvs(a = (0 - 0.13)/0.07, b = np.inf, loc=0.13, scale=0.07, size=1))
    pomegab_0 = float(np.random.uniform(0, 2*np.pi, size=1))

    thetab_0 = float(np.random.uniform(0, 2*np.pi, size=1))

    deltaT_0 = float(rng.uniform(0, 2000, size=1))

    prior_set = (eforced_0, efree_0, mu_0, deltaT_0, mb_0, eb_0, pomegab_0, thetab_0)

    return prior_set

def compute_model(theta):

    sim = makesim(theta)
    pratios_cd, pratios_db = run(sim)
    return pratios_cd[obs_idx], pratios_db[obs_idx]



def lnlike(theta):

    model_cd, model_db = compute_model(theta)
    argument_cd, argument_db = (obs - model_cd)**2 / obs_err**2, (obs_db - model_db)**2 / obs_err_db**2
    loglike_cd, loglike_db = 0.5*np.sum(argument_cd), 0.5*np.sum(argument_db)
    return loglike_cd, loglike_db

def get_posteriors(seed):

    prior = gen_priors_array(seed)
    like_val = lnlike(prior)

    rng = np.random.default_rng(seed)

    prob = np.log10(np.random.random())


    if -like_val[0] > prob and -like_val[1] > prob:
        post = prior

        return post

    else:
    	return np.nan

################################################################################

seeds = list(range(int(13000000), int(1e8)))
#prior_test = np.array([gen_priors_array(seed) for seed in seeds])
#num_cores = multiprocessing.cpu_count()
#start_time = timeit.default_timer()

n_batch = int(1e5)
seed_batches = [seeds[i:i + n_batch] for i in range(0, len(seeds), n_batch)]
# start_time = timeit.default_timer()
path = '/Users/Helios/gdrive_pu/tamayo_research/lnlike_planetb_pratio/batch_{}.npy'
for i,seed_batch in enumerate(seed_batches):
    if __name__ == '__main__':
        #pool = Pool()
        #results = pool.map(get_posteriors,priors)
        results = p_map(get_posteriors,seed_batch)
        #pool.close()
        #pool.join()
        np.save(path.format(i+130), results) # 25 mil broke after file 257 so replacing format so it doesn't overwrite
        #print("--- %s seconds ---" % (timeit.default_timer() - start_time))

        print(len(np.array(results)[pd.notnull(np.array(results))]))
