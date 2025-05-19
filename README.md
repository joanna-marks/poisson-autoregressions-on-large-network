# Poisson Autoregression on Large Network

To run a simulation with given paramteres (final time, kernel combinations, probability matrix, community sizes) set the desired combination of parameters in src/config.py file then run the main.py file. After running the main.py file copy the path of to which the results have been saved (will be printed at the end of simulation), paste it into the plots.py file and run that to produce plots of lambdas over time and save it to the given path.

If you want to create a plot of the error dependence on N run the N_dependence.py file.

Results from the previous runs can be found in the results folder (the name of the folder record what parameters were used, all the information about the run  i.e. all the  hyperparameters can be found in the metadata.py file in the given folder)