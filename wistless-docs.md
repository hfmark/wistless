# Basic usage of wistless
`wistless` is an interface to a database of perple\_X calculations for rock samples where users can filter the database and visualize different sets of parameters.

Users start by setting up filter conditions. Conditions can be set up based on bulk composition of samples, rock type, state variables like P and T, and/or calculated quantities like seismic velocities. Multiple filter conditions can be applied simultaneously. 

When the query is run, `wistless` filters its database and returns user-specified quantities for samples at specific points in parameter space which meet all of the conditions.

***

## What is in the database?
The database contains information for lower crustal rock samples run through perple\_X, from the Hacker et al. (2015) compilation. Each sample has an associated bulk composition, rock type, wt%SiO2, Mg#, and calculated quantities (vp, vs, density) over a grid of state variables (P and T). 

Queries to the database return information for points in the database that satisfy all the filter conditions. This means that queries do not return a list of rock samples; rather, they return sample parameters at particular state variable conditions. If a rock sample matches the filter conditions at multiple points in (P, T) space, that sample will appear multiple times in the results. Users can limit the state variable space using filters, and can even specify a single point in state variable space if they want.

### Caution: the database is large
The Hacker et al. (2015) compilation contains ~5500 rocks. For each of these, we have calculated quantities over a 100x100 grid of P and T. This gives a table with ~5e7 individual entries. Broad filters (for example, all samples with Vp > 4) will therefore return a *lot* of entries and may crash the interface. If you need to run very broad queries, contact the maintainers for information on how to download the database and query it locally.

## How to set up filters
Pretty much all values can be used to filter the database. Standard conditional filters (<, >, =) are available.

The most useful filter types are usually *+-*, where the user sets a center value and an absolute range; and *%*, where the user sets a center value and percentage range. 

The *in* option lets users select a specific range for a quantity within the full range of values in the database. For bulk composition this may be helpful; for calculated quantities like vp and vs, there are some extreme outlier values that make the sliders less useful.

## Return quantities
Users specify which quantities to return. If return quantities are not selected, the query will just return a database-internal ID number for each matching sample/point, which is not very useful. The list of available quantities is the same as the list of things that you can filter on: bulk composition, rock type, state variables, calculated quantities. 

The quantities selected to return are the quantities that can be output to csv, and that are available for plotting in the **data viz** tab.

## Plotting options
After running a query, users can download the results as a csv file and analyze/plot whatever they like. 

`wistless` also provides some basic data visualization options in the **data viz** tab. Options include making 2D scatter plots of different pairs of quantities; histograms; a heatmap showing the density of results returned in P/T space; and pie charts for discrete categorical variables. These plots use the `altair` python library.

## Fitting and other calculations
The **(mis)fitting** tab provides some basic functions for calculating misfit, and fitting returned quantities.

The 'calculate (joint) misfit' button calculates the misfit for Vp, Vs, and/or Vp/Vs for the results. This calculation only works if at least one of Vp, Vs, and Vp/Vs was used for a filter condition; and if that quantity is included in the returned results. The joint misfit is the root square sum of all of the Vp, Vs, and/or Vp/Vs misfits.

The 'calculate best fit T' button calculates a best-fitting temperature from the results provided temperature has been included in the returned results.

If pressure and temperature are both included in the returned results, and the joint misfit has been calculated, an option to calculate the misfit-weighted mean and standard deviation of other returned quantities will appear. To do this, select which returned (numeric) property you want to fit, and select the P and T conditions at which to fit. The sliders for P and T will only assume discrete values that are present in the returned results. If there are enough returned points at the selected (P, T) point, pressing the 'fit property' button will print the misfit-weighted mean, mifit-weighted standard deviation, and number of points included in the fitting calculation.

