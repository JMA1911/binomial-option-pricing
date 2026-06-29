# Binomial Option Pricing

## MSc Financial Mathematics – Computational Finance Project (University of Leeds)

This repository contains the Python implementation and report for my MSc Financial Mathematics Computational Finance project completed at the University of Leeds.

The project implements and compares binomial tree methods for pricing European call and put options, with a focus on computational efficiency, convergence behaviour and validation against the Black-Scholes model.

---

## Project Overview

The analysis begins with a standard Cox-Ross-Rubinstein (CRR) binomial tree implementation before developing an optimised no-array version to reduce memory usage.

The project then compares several numerical pricing approaches:

- Standard CRR binomial tree
- Optimised no-array CRR implementation
- Exact CRR method
- Tilted tree method
- Richardson extrapolation
- Black-Scholes benchmark pricing

---

## Key Features

- European call and put option pricing
- Computational complexity comparison
- Optimised no-array implementation
- Black-Scholes validation
- Convergence analysis
- Stock price distribution comparison
- Tilted tree implementation
- Richardson extrapolation

---

## Repository Contents

- `binomial_option_pricing.py` – Python implementation
- `Binomial Option Pricing Report.pdf` – Technical report describing the methodology, implementation and results

---

## Technologies

- Python
- NumPy
- SciPy
- Matplotlib
- Pandas

---

## Key Results

- Improved memory complexity from O(M²) to O(1) using an optimised no-array implementation.
- Verified binomial option prices against Black-Scholes benchmark prices.
- Demonstrated convergence of binomial CRR prices to Black-Scholes values as the number of time steps increases.
- Implemented tilted trees and Richardson extrapolation to analyse convergence behaviour and pricing accuracy.

---

*Academic coursework completed as part of the MSc Financial Mathematics programme at the University of Leeds.*
