# ExoSim

ExoSim (Exoplanet Observation Simulator) is a generic time-domain simulator of exoplanet transit spectroscopy.

Installation
------
We recommend setting up a virtual environment  for ExoSim to avoid package conflicts.  Using conda the following command line instruction will do all this for you producing a stable environment for this version:

    conda create -n exosim python=3.8.5 matplotlib=3.3.1-0 setuptools=49.6.0 numpy=1.19.1

Then activate this environment. Depending on the system the activation command may be any one of the following:

    source activate exosim
    
or    

    conda activate exosim
    
or    
    
    activate exosim

There is currently no setup.py file for this version so please install the following additional packages using pip inside your virtual environment.
    
    pip install pytransit==2.1.1
    pip install scipy==1.5.2
    pip install astropy==4.0.1
    pip install quantities==0.12.4
    pip install emcee==3.0.2
    pip install seaborn==0.10.1
    pip install uncertainties==3.1.4
    pip install tqdm==4.48.2
    pip install lxml==4.5.2
    pip install photutils==1.0.1
    pip install exodata
    
    
### GitHub

Next, download the ExoSim repository from github and unzip on your computer.

### Databases

Next, download the following databases.  

[Phoenix BT-Settl database](https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011_2015/FITS/BT-Settl_M-0.0a+0.0.tar) (Allard F., Homeier D., Freytag B., 2012, Philos. Trans. Royal Soc. A, 370, 2765).  Then move the folder into `exosim/data/` .  
In the input configuration file `exosim_defaults.xml` change the `StarSEDPath` val entry to val= "your path to the folder".

[Open Exoplanet Catalogue](https://github.com/OpenExoplanetCatalogue/open_exoplanet_catalogue/). Then move the folder into `exosim/data/` .
In the input configuration file `exosim_defaults.xml` change the `OpenExoplanetCatalogue` val entry to val= "your path to the systems folder".

### Running a simulation

If running from the terminal you will need to add exosim to the PYTHONPATH

     PYTHONPATH=$PYTHONPATH:/....your path to.../ExoSimPublic-master 
     export PYTHONPATH

ExoSim should now be ready to run. Run from the terminal with:

      python runexosim.py
      
FITS files with the image time series are deposited in the folder `ExoSimOutput` which will appear in your home directory.


Citing
------

If you use ExoSim in your research, please cite:
Sarkar, S., Pascale, E., Papageorgiou, A. et al. ExoSim: the Exoplanet Observation Simulator. Exp Astron (2021). https://doi.org/10.1007/s10686-020-09690-9





