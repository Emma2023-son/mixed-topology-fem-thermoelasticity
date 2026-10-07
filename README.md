# Mixed-Topology FEM for Thermoelasticity

Reproducibility package for the numerical assessment of mixed-topology conforming finite elements for thermoelasticity on an L-shaped domain.

## Overview

This repository contains the Python implementation and numerical outputs used in the study:

> **Numerical Assessment of Mixed-Topology Conforming Finite Elements for Thermoelasticity on an L-Shaped Domain**

The study evaluates conforming finite-element discretisations based on linear triangular elements (T3), bilinear quadrilateral elements (Q4), and a mixed T3–Q4 topology for coupled thermoelastic problems on an L-shaped domain.

The computational framework is designed to assess:

- numerical convergence;
- displacement and strain accuracy;
- energy-norm behaviour;
- mesh and degree-of-freedom requirements;
- computational cost;
- finite-mesh stress scaling near the re-entrant corner;
- independent manufactured-solution verification; and
- the effect of Q4 localisation within the mixed topology.

The repository is intended to provide a transparent and reproducible computational record of the numerical experiments reported in the manuscript.

---

## Computational framework

The implementation uses Python to perform:

1. mesh generation for the L-shaped domain;
2. T3, Q4, and mixed T3–Q4 discretisation;
3. thermal finite-element assembly;
4. plane-stress thermoelasticity assembly;
5. solution of the coupled thermoelastic problem;
6. stress recovery;
7. mesh-refinement studies;
8. manufactured-solution verification;
9. finite-mesh corner-stress scaling;
10. matched-mesh convergence and cost assessment; and
11. controlled Q4-localisation experiments.

All meshes and numerical inputs required by the reported experiments are generated programmatically by the released scripts.

---

## Repository structure

```text
mixed-topology-fem-thermoelasticity/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
│
├── src/
│   └── validation.py
│
├── scripts/
│   ├── annular_singularity.py
│   ├── matched_dof_cost_comparison.py
│   ├── mesh_dof_verification.py
│   
│
├── results/
│   ├── validation_results/
│   ├── annular_singularity_results/
│   └── Dof_cost_comparison_results/
│
├── figures/
│   ├── validation_figures/
│   ├── annular_singularity_figures/
│   └── Dof_cost_comparison_figures/
│
└── manuscript/
    └── tables/
