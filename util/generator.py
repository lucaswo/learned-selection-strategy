import numpy as np
from itertools import product

# N buckets
# k max elements per bucket (usually 100 for percent)
# delta diff for each step
def generate_bw_hist(generator: str, N: int = 64, k: int = 100, bit: bool = False) -> None:
    if generator == "laola":
        syn_data = gen_laola(N, k)
    elif generator == "outlier":
        syn_data = gen_outlier(N, k)
    elif generator == "tidal":
        syn_data = gen_tidal(N, k)
    else:
        raise Exception("Generator not implemented.")

    if not bit:
        header = ["bwHist_{}".format(x) for x in range(1,N+1)]
        header =  "\t".join(header)
        np.savetxt("measurements_hist/train_{}.tsv".format(generator), syn_data, delimiter="\t", fmt='%i', header=header, comments="")
    else:
        with open("measurements_hist/train_{}.bin".format(generator), "wb") as outFile:
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