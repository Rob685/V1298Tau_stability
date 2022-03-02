# V1298Tau_stability
Repository for V1298Tau stability constraints

`Figures.ipynb` shows all of the process of the project and produces all the figures. This notebook contains the `sampler` function, which samples the RV masses from suárez Mascareño et. al (2021; https://ui.adsabs.harvard.edu/abs/2021NatAs...6..232S/abstract), under the assumption that such posteriors are Gaussian.

`ln_like_sampler.py` is the rejection sampling script. For reproducibility, the rejection process generates priors given a random seed. In the terminal, a user needs to input the range of seeds, which is equivalent to the range of samples; e.g, `$ python3 ln_like_sampler.py 0 30000000` for 30 million samples. The process saves a file into a directory with the posteriors every 10,000 samples. 

The data folder contains the rejection posteriors and the stability posteriors, named by the number of samples drawn and which priors on the mass of planet b we used. The stability posteriors folder also contains the SPOCK probabilities for each of the configurations since these take hours to compute.

The hamiltonian files (`deltaH_cd.npy`, etc.) and resonance angle files take a while to compute, so those files are also available in the stability posteriors folder. 
