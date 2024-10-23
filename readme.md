## WISTLESS

WISTLESS (Whole-rock Interpretive Seismic Toolbox for LowEr cruStal Samples) is a GUI interface for querying a database of lower crustal samples. Seismic properties have been calculated for each sample over a grid of relevant P and T conditions using Gibbs free energy minimization (via perpleX). The GUI is a streamlit app which enables users to construct SQL queries, run them, export the results, and do some basic data visualization.

The app requires the following (python) dependencies:
- python 3.10+ (or higher? check this)
- numpy
- pandas
- duckdb
- altair
- streamlit
The recommended way to set up dependencies is in an isolated environment, using a tool like conda. For example: `conda create --name wistless numpy pandas python-duckdb altair-all` will install most of the dependencies; if you then activate the environment with `conda activate wistless`, you can install streamlit with `pip install streamlit` (streamlit is not available in the conda package repo).

The command to run the app is `streamlit run main.py`.
