# NS3-RL_AUV
## Documents

https://docs.google.com/document/d/13UO1IhOEJdCyAtrmCC8jLPergnCrxMsh0f-RNLrHes4/edit?usp=sharing

## Install 
### Install packages library 
```
pip3 install torch, matplotlib, numpy, pandas, seaborn
```

### Install NS3
```
wget https://www.nsnam.org/releases/ns-allinone-3.32.tar.bz2 
tar -xvf /content/ns-allinone-3.32.tar.bz2
cd /ns-allinone-3.32/ns-3.32
./waf configure
./waf
```

## Run
```
cd folder_have_waf_file 
./waf --pyrun scrach/uan_simulator.py
```
## Note 
```
Note that install package for Python bindings in NS3
```
