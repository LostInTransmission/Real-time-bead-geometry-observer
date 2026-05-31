# Real-time-bead-geometry-observer
Official code for the article "Real-time geometric observation from noisy laser profilometry: comparative benchmarking of filtering methods for closed-loop additive manufacturing control systems" by Timofejs Oniscuks.

# How to work with the files:
In the releases - complete raw poin log for the 30 layer structure with ULS = 0.8 and TS = 15mm/s.
In the data folder - .csv log file of single thin-wall structure layer.
It can be used to simulate the work of the pipeline by running the Simulator file from src folder. The src folder as well includes 3 pipeline files - 1 for each different filter (SOR, Median, DBSCAN). To work with specifik filter the Simulator file must be changet (the variable at the beginning, chosing the filter).
The Simulator file would create 2 .csv files in the Results folder - _Apexes.csv and _CalculatedHeights.csv.

In the benchmark folder the logic of the used metrics is demonstrated (Spikes, FFT, Lag-1). It can be experimented with by running the Benchmark Runner.py from the same folder. Benchmark Runner.py works with the _Apexes file for chosen layer (variable in the code file beginning). 

