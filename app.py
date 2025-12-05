# app.py
import streamlit as st
import numpy as np
from ssa import SalpSwarmAlgorithm

def sphere(x):
    return sum(x**2)

st.title("Salp Swarm Algorithm (SSA) Optimization Demo")

dim = st.slider("Dimension of problem", 1, 10, 5)
num_salp = st.slider("Number of salps", 10, 100, 30)
max_iter = st.slider("Maximum iterations", 10, 200, 50)

if st.button("Run SSA"):
    ssa = SalpSwarmAlgorithm(sphere, dim=dim, num_salp=num_salp, max_iter=max_iter)
    best_pos, best_fit = ssa.optimize()
    st.write("Best position found:", best_pos)
    st.write("Best fitness:", best_fit)
