# Learned Selection Strategy for Integer Compression

A local Selection Strategy for Integer Compression algorithms as presented in [[1]](#references).

### Install
Get all required packages with (make sure that you are using Python 3.10 or above):

```bash
pip3 install -r requirements.txt
```

### Usage
The `learned-selection-strategy.py` script can be executed via
```bash
python3 learned-selection-strategy.py [options]
```

Get all options by calling `python3 learned-selection-strategy.py -h`

```
>>> python3 learned-selection-strategy.py -h
usage: learned-selection-strategy.py [-h] [-g GENERATOR] [-d]

A local learned selection strategy for lightweight integer compression

options:
  -h, --help            show this help message and exit
  -g GENERATOR, --generator GENERATOR
                        Which generator to be used. Can be laola, outlier, or tidal. Default: laola
  -d, --datagen         If set, only the data generation is executed.
```

## References

[1] [Woltmann et al., Learned Selection Strategy for Integer Compression, EDBT 2023](https://dx.doi.org/10.48786/edbt.2023.47)

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{woltmann2023learned, 
  doi = {10.48786/edbt.2023.47}, 
  url = {https://openproceedings.org/2023/conf/edbt/3-paper-48.pdf}, 
  author = {Woltmann, Lucas and Damme, Patrick and Hartmann, Claudio and Habich, Dirk and Lehner, Wolfgang}, 
  language = {en}, 
  title = {{Learned Selection Strategy for Lightweight Integer Compression Algorithms}}, 
  publisher = {OpenProceedings.org}, 
  year = {2023}, 
  booktitle = {Proceedings of the 26th International Conference on Extending Database Technology}, 
  location = {Ioannina, Greece},
  series = {EDBT 2023}} 
```