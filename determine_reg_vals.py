# determining lambda 1 and 2 for the quad. ROM case:

import opinf
import diffrax
import jax.numpy as jnp
import numpy as np
import pickle

def ckron_numpy(arr):
    return np.concatenate([arr[i] * arr[: i + 1] for i in range(arr.shape[0])], axis = 0)

def ckron_jnp(arr):
    return jnp.concatenate([arr[i] * arr[: i + 1] for i in range(arr.shape[0])], axis = 0)

import numpy as np
import math
import matplotlib.pyplot as plt
from jax import grad, vmap

def s_jax(c, mu, x, t):
    const = 2e-4
    return (1 / jnp.sqrt(const * jnp.pi)) * jnp.exp(-((x - c * t - mu) ** 2) / const)

ds_dt = vmap(grad(s_jax, argnums = 3), (None, None, 0, None)) 

x_vals = np.linspace(0, 1, 2**12)
t_vals = np.linspace(0, 0.1, 2000)  
c_val = 10
mu_val = 0.1

solution = []
solution_der = []

for i in range(len(t_vals)):
    t_val = t_vals[i]
    solution.append( np.array(s_jax(c_val, mu_val, x_vals, t_val) ))
    solution_der.append( np.array(ds_dt(c_val, mu_val, x_vals, t_val)) )

S_train = np.array(solution).T #(x,t)
S_train_dot = np.array(solution_der).T #(x,t)
S_train_ref = np.mean(S_train, axis = 1)
S_train_cent = S_train - S_train_ref.reshape(-1,1).repeat(S_train.shape[1], axis = 1)


import itertools
trunc_dim = 29
d_val = 1 + trunc_dim + int(trunc_dim * (trunc_dim + 1)/2)
lambda_1, lambda_2 = np.logspace(3, 11, 10), np.logspace(3,11, 10)
hyperparam_values = [val for val in itertools.product(lambda_1, lambda_2)]
reg_vals = np.zeros((len(hyperparam_values),  d_val))

for i in range(reg_vals.shape[0]):
  reg_vals[i,:trunc_dim+1] = np.sqrt(hyperparam_values[i][0]) * np.ones(trunc_dim +1)
  reg_vals[i,trunc_dim+1:] = np.sqrt(hyperparam_values[i][1]) * np.ones(d_val - (trunc_dim+1))

# print(reg_vals[-1])

comb_err_vals = {}

c_val_test = 10
mu_val_test = 0.12547
t_vals_test = np.arange(0, 0.08+1e-6, 1e-6)
# exact FOM:
solution = []
solution_der = []

for i in range(len(t_vals_test)):
    t_val = t_vals_test[i]
    solution.append( np.array(s_jax(c_val_test, mu_val_test, x_vals, t_val) ))
    solution_der.append( np.array(ds_dt(c_val_test, mu_val_test, x_vals, t_val)) )

S_test = np.array(solution).T #(x,t)

for i in range(reg_vals.shape[0]):

    quadratic_rom = opinf.ROM(
        basis = opinf.basis.PODBasis(num_vectors = 29),
        model = opinf.models.ContinuousModel(
            operators='cAH',
        solver = opinf.lstsq.TikhonovSolver(regularizer = reg_vals[i])
        )
    )
    quadratic_rom.fit(S_train_cent, S_train_dot)

    s0 = s_jax(c_val_test, mu_val_test, x_vals, t_vals_test[0])
    s0 = s0 - S_train_ref
    s0 = quadratic_rom.basis.compress(s0)

    S_pred = quadratic_rom.model.predict(s0, t_vals_test, method = "BDF", max_step = t_vals_test[1] - t_vals_test[0])

    if(S_pred.shape[1] < S_test.shape[1]):
        continue

    S_pred = quadratic_rom.basis.decompress(S_pred)

    S_prediction_1 = S_pred + S_train_ref.reshape(-1,1).repeat(S_pred.shape[1], axis = 1)

    comb_err_vals[tuple([reg_vals[i][0], reg_vals[i][-1]])] = np.linalg.norm(S_prediction_1 - S_test, 'fro') # lambda1 and lambda2 tuple.


with open("comb_err_vals.pickle", "wb") as file:
    pickle.dump(comb_err_vals, file)


file = open('comb_err_vals.pickle', 'rb')

data = pickle.load(file)
min_key = min(data, key=data.get)

# Get the minimum value
min_value = data[min_key]

print("Key with minimum value:", min_key)
print("Minimum value:", min_value)
