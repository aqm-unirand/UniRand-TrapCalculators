# FermiQP-physics

## Installation

NOTE: for using the streamlit, you need to first install the fermiqp-style package


Preferred:
```sh
python -m pip install git+ssh://git@gitlab.mpcdf.mpg.de/mpq-fermiqp/fermiqp-physics
```

## Streamlit

To start app, you can run the following in the environment with package installed:

```sh
fermiqp-app
```

## Important changes

- use lattice.py to replace old opticallattice.py for simplicity ("opticallattice.xxx" still works, just replace previous "lattice.opticallattice" by "lattice")
- add experiments.py as a higher-level implementation: add pinning lattice & geometric beam
- add pre-calculated Zeeman shifts for calculating RF, MW and closed optical transitions
