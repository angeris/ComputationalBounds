# Code for Computational Bounds for Photonic Design

This repository contains the code used to generate the images for the paper
[Computational Bounds for Photonic Design](https://arxiv.org/abs/1811.12936) by
Guillermo Angeris, Jelena Vučković, and Stephen Boyd.

## Running the script
Make sure you have [all of the requirements](##Requirements).

Clone the repository and simply run (assuming that [Julia](https://julialang.org) is found in your `PATH`)
```bash
julia generate_figures.jl
```
The script will then run (producing a good amount of output in the console, for those interested) for a few minutes on most modern laptops.
It should then produce several `*.pdf` files in the directory that contains it.

### Changing parameters
There are a few parameters to change. The ones currently set are exactly the ones provided in the [original paper](https://arxiv.org/abs/1811.12936).

See the code for further documentation.

## Requirements
This package has the following requirements:
- PyPlot
- JuMP
- ProgressMeter

And, if you plan to use it directly, it will also require
- Mosek
- MosekTools

Additionally, if you'd like to save the iterates, you will also need
- JLD

You can install any of these from the Julia console (REPL) by typing `]` and
`add [PACKAGE_NAME]`. See the [Julia documentation](https://docs.julialang.org/en/v1/stdlib/Pkg/index.html) for more details.