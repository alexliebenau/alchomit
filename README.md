# alchomit Optimization Library
**AL**gorithms for <br />
**C**omplex<br /> 
**H**igh-dimensional<br />
**O**ptimization and<br />
**M**ixed<br />
**I**nteger<br />
**T**ransformation<br />

This is an optimization library intended for large-scale mixed-integer optimization. It has been developed for the 
optimization of Modelica energy system models, however it can be used for other purposes with some tweaking.

Please note that this library is in its early stages and subject of further development. The current functionality has 
however been tested and verified.

## General

Here's a brief overview of the current functionality:
- Interface for Modelica simulations
- Parameter optimization, either global or for each time step (control)
  - Includes the following algorithms:
    - BFGS
    - Rand Search, Grid Search, Staged Search (Blend of random and grid)
    - Evolutionary Algorithm
    - Bayesian Optimization
    - more to come in future release...

Please refer to the documentation file 'doc.pdf' for insights on how this library works internally.

## Usage

Hint: ReadTheDocs doc is in progress. For now, please look at the 'doc.pdf' and the provided examples.

1. Define *variables*
2. Define *constraints*
3. Initialize *framework*
4. Create *model* and *optimizer/controller* instances
5. Run optimization

## Planned features

- Include support for Matlab and Excel
- Improve multithreading