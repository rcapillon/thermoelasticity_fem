# thermoelasticity_fem
A Finite Element solver for classical linear thermoelasticity in Python

**Important note**: This repository has just recently been made public. It is uncommented, without documentation and 
possibly still bugged. For now, users are incited to read the example scripts in order to make their own. There is a 
steady-state and a transient example, showcasing respectively the two available solvers.

## Bibliography
[1] Balla, M. “Formulation of coupled problems of thermoelasticity by finite elements", 
Periodica Polytechnica Mechanical Engineering, 1989.
[2] Selverian, J., Johnston, D., Johnson, M. A., Fox, K. M., & Hellmann, J. R.,
"Temperature dependent elastic properties of several commercial glasses", 
High Temperatures - High Pressures, 2010.
[3] Colinas-Armijo, N., Inigo, B., Aguirre, G., 
"Comparison of Thermal Modal Analysis and Proper Orthogonal Decomposition methods for thermal error estimation", 
Special Interest Group Meeting on Thermal Issues, 2020. 

Reference [1] was used for the equations of classical linear thermoelasticity. Reference [2] was used to obtain
reasonable properties for a glass material. Reference [3] is meant to be used for reduced-order modeling.