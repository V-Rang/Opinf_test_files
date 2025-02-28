# determining lambda 1 and 2 for the quad. ROM case:

import opinf
import diffrax
import jax.numpy as jnp
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
from jax import grad, vmap

def ckron_jnp(arr):
    return jnp.concatenate([arr[i] * arr[: i + 1] for i in range(arr.shape[0])], axis = 0)

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

# rom to be fit b/w S_train_cent and S_train_dot:
quad_reg = np.zeros((int(1 + 29 + (29*30)/2)))
# 1 + r rows corresp. to lambda1
# remaining r(r+1)/2 corresp. to lambda2.
lambda_1, lambda_2 = (87.9922543569107)**2,  (1895.7356524063753)**2
trunc_dim = 29
d_val = 1 + trunc_dim + int(trunc_dim * (trunc_dim + 1)/2)
reg_vals = np.zeros((d_val))
reg_vals[:trunc_dim+1] = np.sqrt(lambda_1) * np.ones(trunc_dim +1)
reg_vals[trunc_dim+1:] = np.sqrt(lambda_2) * np.ones(d_val - (trunc_dim+1))


quadratic_rom = opinf.ROM(
    basis = opinf.basis.PODBasis(num_vectors = 29),
    model = opinf.models.ContinuousModel(
        operators='cAH',
    solver = opinf.lstsq.TikhonovSolver(regularizer = reg_vals)
    )
)

quadratic_rom.fit(S_train_cent, S_train_dot)
c_val_test = 10
mu_val_test = 0.12547
t_vals_test = np.arange(0, 0.01+1e-6, 1e-6)

# exact FOM:
solution = []
solution_der = []

for i in range(len(t_vals_test)):
    t_val = t_vals_test[i]
    solution.append( np.array(s_jax(c_val_test, mu_val_test, x_vals, t_val) ))
    solution_der.append( np.array(ds_dt(c_val_test, mu_val_test, x_vals, t_val)) )

S_test = np.array(solution).T #(x,t)

#prediction using BDF.
s0 = s_jax(c_val_test, mu_val_test, x_vals, t_vals_test[0])
s0 = s0 - S_train_ref
s0 = quadratic_rom.basis.compress(s0)
s0 = quadratic_rom.model.predict(s0, t_vals_test, method = "BDF", max_step = t_vals_test[1] - t_vals_test[0])
s0 = quadratic_rom.basis.decompress(s0)
S_prediction_1 = s0 + S_train_ref.reshape(-1,1).repeat(s0.shape[1], axis = 1)

# prediction using semi-Implicit Euler:
c_hat_quadratic, A_hat_quadratic, H_hat_quadratic = quadratic_rom.model.operators

c_hat_quadratic = jnp.array(c_hat_quadratic.entries)
A_hat_quadratic = jnp.array(A_hat_quadratic.entries)
H_hat_quadratic = jnp.array(H_hat_quadratic.entries)

# using Semi-implicit Euler:
def implicit_part_quadratic(t, s, args):
    """Stiff part: A s"""
    return A_hat_quadratic @ s

def explicit_part_quadratic(t, s, args):
    """Non-stiff part: c + H[s ⊗ s]"""
    return c_hat_quadratic + H_hat_quadratic @ ckron_jnp(s)

stepsize_controller = diffrax.PIDController(rtol=1e-2, atol=1e-3)
solver = diffrax.Sil3()

# Define the ODE term (IMEX requires separate explicit & implicit parts)
term_quadratic = diffrax.MultiTerm(
    diffrax.ODETerm(explicit_part_quadratic),  # Explicit
    diffrax.ODETerm(implicit_part_quadratic)   # Implicit
)

t_vals_test = jnp.array(t_vals_test) 
dt_test = t_vals_test[1] - t_vals_test[0]
s_init = s_jax(c_val_test, mu_val_test, x_vals, t_vals_test[0])
s_init = s_init - S_train_ref
s_init = quadratic_rom.basis.compress(s_init)
s_test_0_hat = jnp.array(s_init)

solution_quadratic = diffrax.diffeqsolve(
    term_quadratic,
    solver,
    t0=t_vals_test[0],
    t1=t_vals_test[-1],
    dt0=dt_test,
    y0=s_test_0_hat,    
    stepsize_controller=stepsize_controller,
    max_steps=500000,  # Increase max_steps to allow more solver steps
    saveat = diffrax.SaveAt(ts = t_vals_test)
)

S_quad_predict = solution_quadratic.ys
S_quad_predict = S_quad_predict.T
print(S_quad_predict.shape)
S_quad_predict = quadratic_rom.basis.decompress(S_quad_predict)
S_quad_predict = S_quad_predict + S_train_ref.reshape(-1,1).repeat(S_quad_predict.shape[1], axis = 1)


results = {
    'FOM':  S_test,
    'pred_1': S_prediction_1,
    'pred_2': S_quad_predict
}

import pickle

with open("results_quad_reg.pickle", "wb") as file:
    pickle.dump(results, file)


file = open('results_quad_reg.pickle', 'rb')

data = pickle.load(file)
S_test = data['FOM']
pred_1 = data['pred_1']
pred_2 = data['pred_2']

x_vals = np.linspace(0, 1, 2**12)
test_index = 1000
plt.plot(x_vals, S_test[:,test_index], label = 'exact')
plt.plot(x_vals, pred_1[:,test_index], label = 'quadratic_prediction_1')
plt.plot(x_vals, pred_2[:,test_index], label = 'quadratic_prediction_2')
plt.legend()
plt.savefig('test_compare_quad3.png')
plt.close()
