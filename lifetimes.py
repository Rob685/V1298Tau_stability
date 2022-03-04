import numpy as np
from celmech import Andoyer
import rebound
from celmech.andoyer import get_Xstarres, get_Xstarunstable, get_Xstarnonres, get_Xsep
import pdb
import corner
from tqdm import tqdm
from celmech.andoyer import get_Hsep
import spock
from spock import FeatureClassifier
from scipy import stats
from p_tqdm import p_map

# creates 3-planet sim
def makesim2(param_cd, param_b, seed, dt=None):
    sim = makesimcd(param_cd[:4], dt=dt)

    m_b = 10**param_b[0]
    e_b = param_b[1]
    np.random.seed(seed)
    pomega_b = float(np.random.uniform(0, 2*np.pi, size=1))
    theta_b = float(np.random.uniform(0, 2*np.pi, size=1))
    sim.add(m=m_b, P=sim.particles[2].P*1.946, e=e_b, pomega=pomega_b, theta=theta_b)
    sim.move_to_com()
    return sim

# creates 2-planet model sim
def makesimcd(param_cd,dt=None):
    e_forced, e_free, mu, deltaT = param_cd
    if dt:
        deltaT = dt
    mratio=0.5
    e_com = 0.0

    Mstar = 1.1
    m1 = mratio*10**mu
    m2 = (1-mratio)*10**mu
    phi = np.pi # where equilibrium is
    theta1 = np.pi # so they start on opposite sides

    andvars = Andoyer.from_Z(j=3, k=1, Z=(e_forced+e_free)/np.sqrt(2), phi=phi,
                             Zstar=e_forced/np.sqrt(2), Mstar=Mstar, m1=m1, m2=m2,
                             Zcom=e_com/np.sqrt(2), phiZcom=0, theta1=theta1)

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


rv_post = np.load('stability_posteriors/100k_rv_priors_60mil_lnlike_randomtheta.npy')
preds_rv = np.load('stability_posteriors/100k_rv_stability_preds_randomtheta.npy')

rv_stable = rv_post[preds_rv != 0.0]
preds_stable = preds_rv[preds_rv != 0.0]

# finding the index and probability where the sum is a fifth of the total sum
thresh_index = []
for i,pred in enumerate(preds_stable):
    s = np.sum(np.sort(preds_stable)[:i])
    if s <= np.sum(preds_stable)/5:
        continue
    else:
        print(s, i)
        thresh_index.append(i)
        break

# stability threshold
fifth_thresh = np.sort(preds_stable)[int(thresh_index)]

# configurations with probabilities below this threshold
rv_fifth = rv_stable[preds_stable < fifth_thresh]
# indices of configs below the threshold
fifth_idx = np.where(preds_stable < fifth_thresh)[0]

# parameters for sims
m_b_rv = [param[-2] for param in rv_fifth]
e_b_rv = [param[-1] for param in rv_fifth]

params_cd = [list(rv_fifth[i][:4]) for i in range(len(rv_fifth))]
params_b = [[m_b, e_b] for m_b, e_b in zip(m_b_rv, e_b_rv)]

# creating sims for each config below the threshold
sim_list = []
pbar = tqdm(total=len(rv_fifth))
for i,(pcd,pb) in enumerate(zip(params_cd, params_b)):
    sim = makesim2(pcd, pb, seed=fifth_idx[i])
    sim_list.append(sim)
    pbar.update()
pbar.close()

# time to parallelize
deep_regressor = spock.DeepRegressor(cuda=False)
for sim in tqdm(sim_list[:10]):
    median, lower, upper = deep_regressor.predict_instability_time(sim)
