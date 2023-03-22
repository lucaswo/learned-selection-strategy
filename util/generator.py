import numpy as np
from itertools import product
import time

# N buckets
# k max elements per bucket (usually 100 for percent)
# delta diff for each step
# size = block size or max array size per training sample
def generate_bw_hist(generator: str, N: int = 64, k: int = 100, size: int = 2048, bit: bool = False) -> None:
    if generator == "laola":
        syn_data = gen_laola(N, k)
    elif generator == "outlier":
        syn_data = gen_outlier(N, k)
    elif generator == "tidal":
        syn_data = gen_tidal(N, k)
    else:
        raise Exception("Generator not implemented.")

    train_data = sample_bw_hist(syn_data, size)
    np.savetxt("measurements_notex/train_{}_notex.tsv".format(generator), train_data, delimiter="\t", fmt='%i', comments="")

    if not bit:
        header = ["bwHist_{}".format(x) for x in range(1,N+1)]
        header =  "\t".join(header)
        np.savetxt("measurements_hist/train_{}_hist.tsv".format(generator), syn_data, delimiter="\t", fmt='%i', header=header, comments="")
    else:
        with open("measurements_hist/train_{}_hist.bin".format(generator), "wb") as outFile:
            outFile.write(syn_data.astype('<B'))

def gen_laola(N: int, k: int, delta: int = 2) -> np.array:
    n_samples = (N-1)*(k//delta)+1

    vec = np.zeros(N)
    vec[0] = k
    dists = np.zeros((n_samples, N), dtype=int)
    dists[0] = vec

    i = 1

    for x in range(N-1):
        for j in range(k//2):
            vec[x] -= 2
            vec[x+1] += 2
            dists[i] = vec
            i += 1

    print("Generated {} histograms.".format(n_samples))
    return dists

def gen_outlier(N: int, k: int, delta: int = 2) -> np.array:
    dists = np.zeros(((N-1)*(k//delta)+1, N), dtype=int)

    i = 0
    for s in range(N-1):
        vec = np.zeros(N)
        
        for x in range(0,k,delta):
            vec[s] = k - x
            vec[N-1] = x
            dists[i] = vec
            i += 1
            
    vec = np.zeros(N)
    vec[-1] = k
    dists[i] = vec
    print("Generated {} histograms.".format(len(dists)))
    return dists

def gen_tidal(N: int, k: int) -> np.array:
    minmax = [(x,y) for x,y in product(range(N), range(N)) if x <= y]

    dists = np.zeros((len(minmax), N), dtype=int)

    i = 0
    for x,y in minmax:
        vec = np.arange(1,y-x+2) ** -1.1
        vec = (vec/vec.sum()*k).astype(int)
        dists[i,x:y+1] = vec
        
        i += 1
    
    print("Generated {} histograms.".format(len(dists)))
    return dists

def sample_bw_hist(hists: np.array, size: int) -> np.array:
    start = time.time()
    rng = np.random.default_rng()

    bits = hists.shape[1]
    hists = hists/hists.sum(axis=1)[:,None]

    bins_upper = 2**np.arange(0,bits)
    bins_upper[-1] = 2**(bits-1)-1

    bins_lower = 2**np.arange(0,bits-1)
    bins_lower = np.pad(bins_lower, (1,0))

    cdf = np.cumsum(hists, axis=1)
    values = rng.random((len(hists), size))
    value_bins = np.array([np.searchsorted(c, v) for c, v in zip(cdf, values)])

    data = np.array([rng.integers(lower,upper) for lower,upper in zip(bins_lower[value_bins], bins_upper[value_bins])])
    
    print("Generated training data in {:.2f}s".format(time.time()-start))

    return data
