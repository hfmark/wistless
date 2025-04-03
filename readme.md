## WISTLESS

`wistless` (Whole-rock Interpretive Seismic Toolbox for LowEr cruStal Samples) is a GUI interface for querying a database of lower crustal samples. Seismic properties have been calculated for each sample over a grid of relevant P and T conditions using Gibbs free energy minimization (via perpleX). The GUI is a streamlit app which enables users to construct SQL queries, run them, export the results, and do some basic data visualization.

### but I don't want a GUI!

The database and functions used by the GUI can also be adapted for use in separate Python scripts. If you want to run a large number of queries (for example, to estimate composition over a 2D or 3D seismic velocity model), you will definitely want a script. An example workflow is provided that can be run either as ordinary python scripts or in jupyter using jupytext (start jupyter lab, right-click on one of the `step*.py` scripts from `wistless`, and select "open with" -> "notebook").

## Installation

The app requires the following (python) dependencies:
- python 3.10+ (or higher? check this)
- numpy
- pandas
- duckdb

In addtion to the base dependencies, the GUI requires:
- altair
- streamlit

The example scripts also require:
- scipy

To run the example workflow scripts in jupyter, you will need:
- jupyter
- jupytext

The recommended way to set up dependencies is in an isolated environment, using a tool like conda. For example: `conda create --name wistless numpy pandas python-duckdb altair-all scipy jupyter jupytext` will install most of the dependencies; if you then activate the environment with `conda activate wistless`, you can install streamlit with `pip install streamlit` (streamlit is not available in the conda package repo).

Once you have downloaded and unpacked this repository (ie using `git clone`) and have also downloaded the database, the command to run the GUI app is `streamlit run main.py`.

## Database

The `wistless` database can be downloaded from Zenodo at [link tba]. To use the GUI, make a `Data/` directory in the directory where the `wistless` source files are, and place the database file in `Data/`.

## References

The sample database draws on work from several papers, including:

Hacker, B. R., Kelemen, P. B., & Behn, M. D. (2015). Continental Lower Crust. Annual Review of Earth and Planetary Sciences, 43(1), 167–205. https://doi.org/10.1146/annurev-earth-050212-124117

Huang, Y., Chubakov, V., Mantovani, F., Rudnick, R. L., & McDonough, W. F. (2013). A reference Earth model for the heat‐producing elements and associated geoneutrino flux. Geochemistry, Geophysics, Geosystems, 14(6), 2003–2029. https://doi.org/10.1002/ggge.20129

Kelemen, P. B., Behn, M. D., & Hacker, B. R. (2025). Average composition and genesis of the lower continental crust. Treatise on Geochemistry, 2, 39-81. https://doi.org/10.1016/B978-0-323-99762-1.00121-2

Shinevar, W. J., Jagoutz, O., & Behn, M. D. (2022). WISTFUL: Whole‐Rock Interpretative Seismic Toolbox for Ultramafic Lithologies. Geochemistry, Geophysics, Geosystems, 23(8). https://doi.org/10.1029/2022GC010329

The perpleX calculations are courtesy of Mark Behn and William Shinevar.
