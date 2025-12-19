import numpy as np
from helper_functions import dirac_delta, padding, phi_param_vector_to_tensor, phi_tensor_to_param_vector


#Funcitons for APPROACH 1

def jumps_per_community_over_time(labs, X):
    labs = np.asarray(labs)
    num_communities = np.max(labs) + 1
    one_hot = np.eye(num_communities)[labs]  # shape (N, num_communities)
    return X @ one_hot  # shape (T, num_communities)


def grad_loglik_intuitive(labs, comm_sizes, X, lambda_bar, grad_lambda_bar):
    """
    Clear, step-by-step computation of:
        ∇_θ ℓ(θ) ≈ Σ_{h,t} [ ( Y_t^h (2 - λ̄_t^h) / Ā^h - n^h ) * ∇_θ λ̄_t^h ]

    Returns
    -------
    grad : (P,) ndarray
    """
    labs = np.asarray(labs)
    X = np.asarray(X)
    lambda_bar = np.asarray(lambda_bar)           # (T, q)
    grad_lambda_bar = np.asarray(grad_lambda_bar) # (T, q, num_params)

    T, N = X.shape
    num_comm = lambda_bar.shape[1]
    num_params = grad_lambda_bar.shape[2]

    # Y_t^h: jumps per community/time
    Y = jumps_per_community_over_time(labs, X)  # (T, q)

    # Ā^h: mean over time of Y_t^h
    Abar = Y.mean(axis=0)/comm_sizes # (T, q)

    grad = np.zeros(num_params, dtype=float)

    # Loops for clarity
    for h in range(num_comm):
        n_h = comm_sizes[h]
        Abar_h = Abar[h]
        for t in range(T):
            Y_t_h = Y[t, h]
            lmb_t_h = lambda_bar[t, h]
            # safe handling when Ā^h == 0
            if Abar_h > 0:
                coeff = Y_t_h * (2.0 - lmb_t_h) / Abar_h - n_h
            else:
                coeff = -n_h
            grad += coeff * grad_lambda_bar[t, h, :]

    return grad


def grad_loglik_fast(labs, comm_sizes, X, lambda_bar, grad_lambda_bar):
    """
    Vectorized version of:
        ∇_θ ℓ(θ) ≈ Σ_{h,t} [ ( Y_t^h (2 - λ̄_t^h) / Ā_t^h - n^h ) * ∇_θ λ̄_t^h ]

    Shapes
    ------
    labs            : (N,)
    comm_sizes      : (q,)
    X               : (T, N)
    lambda_bar      : (T, q)
    grad_lambda_bar : (T, q, P)

    Returns
    -------
    grad : (P,) ndarray
    """

    labs = np.asarray(labs)
    comm_sizes = np.asarray(comm_sizes)
    X = np.asarray(X)
    lambda_bar = np.asarray(lambda_bar)           # (T, q)
    grad_lambda_bar = np.asarray(grad_lambda_bar) # (T, q, P)

    T, N = X.shape
    T_l, q = lambda_bar.shape
    T_g, q_g, P = grad_lambda_bar.shape

    assert T_l == T and T_g == T, "Time dimension T must match."
    assert q_g == q, "Community count q must match."
    assert comm_sizes.shape[0] == q, "comm_sizes must have length q."

    # Y_t^h: jumps per community/time, shape (T, q)
    Y = jumps_per_community_over_time(labs, X)

    # Ā_t^h: your current code uses cumulative sum over time, divided by T
    # (T, q). This preserves the original semantics.
    Abar = Y.mean(axis=0)/comm_sizes

    # Build coeff[t, h] = Y_t^h (2 - λ̄_t^h) / Ā_t^h - n^h  (if Abar_t^h > 0)
    #                   = -n^h                              (otherwise)

    # Broadcast comm_sizes to (T, q)
    n = comm_sizes.reshape(1, q)              # (1, q)
    coeff = -np.broadcast_to(n, Y.shape).astype(float)  # (T, q), start with -n^h

    # Mask where Abar_t^h > 0
    mask = Abar > 0

    # Add the Y*(2-λ)/Abar term only where Abar > 0
    coeff[mask] += (
        Y[mask] * (2.0 - lambda_bar[mask]) / Abar[mask]
    )

    # Now grad = Σ_{t,h} coeff[t,h] * grad_lambda_bar[t,h,:]
    # Use einsum to contract over t and h
    grad = np.einsum('tq,tqp->p', coeff, grad_lambda_bar)

    return grad


def lambda_bar_and_grad_for_all_times(mu,phi, alpha, prob_matrix, comm_sizes, p, T):
    """
    Compute λ̄_t^h and its gradients wrt μ^r and φ^{sr}_m,
    for t = 1,...,T (stored at index t-1).

    Parameters
    ----------
    theta       : (num_comm + num_comm*num_comm*p,) parameter vector
    alpha       : (num_comm,)
    prob_matrix : (num_comm, num_comm)  p_{l -> h}
    comm_sizes  : (num_comm,)  (used only to get num_comm)
    p           : max lag for φ^{hl}_m
    T           : number of time steps

    Returns
    -------
    lambda_bar      : (T, num_comm)
    grad_lambda_bar : (T, num_comm, num_params)
    """

    # number of communities
    num_comm = len(comm_sizes)

    # unpack parameters
    # mu = np.asarray(theta[:num_comm])
    # phi_flat = np.asarray(theta[num_comm:])
    # assert phi_flat.size == num_comm * num_comm * p

    # # φ[h, l, m-1] = φ^{hl}_m
    # phi = phi_flat.reshape(num_comm, num_comm, p)
    # pad_width = ((0, 0), (0, 0), (0, T - p))   # pad only last axis
    # if T > p:
    #     phi = np.pad(phi, pad_width, mode="constant", constant_values=0)

    phi = padding(phi, T)
    alpha = np.asarray(alpha)
    prob_matrix = np.asarray(prob_matrix)

    # parameter layout
    num_mu = num_comm
    num_phi = num_comm * num_comm * p
    num_params = num_mu + num_phi

    # storage for λ̄_t and ∂λ̄_t
    lambda_bar = np.zeros((T, num_comm))
    grad_lambda_bar = np.zeros((T, num_comm, num_params))

    # ---------------------------
    # t = 1 case
    # ---------------------------
    # λ̄_1^h = μ^h
    lambda_bar[0, :] = mu

    # ∂λ̄_1^h / ∂μ^r = δ_{hr}, rest = 0
    for h in range(num_comm):
        grad_lambda_bar[0, h, h] = 1.0

    # ---------------------------
    # t = 2,...,T
    # ---------------------------
    for tau in range(1, T):                # tau = t-1
        # ---------------------------
        # compute λ̄_t^h
        # ---------------------------
        for h in range(num_comm):
            lambda_bar[tau, h] = mu[h] + np.dot(alpha * prob_matrix[:, h], np.sum(phi[:tau, h, :][::-1, :] * lambda_bar[:tau, :] , axis=0))
            # print(lambda_bar[:tau, :])
            # print(phi[:tau, h, :])

    # ---------------------------
        # compute ∂λ̄_t^h / ∂μ^r
        # ---------------------------

            for r in range(num_comm):
                base = dirac_delta(h, r)
                rec = np.dot(alpha * prob_matrix[:, h], np.sum(phi[:tau, h,:][::-1, :] * grad_lambda_bar[:tau, :, r] , axis=0))

                grad_lambda_bar[tau, h, r] = base + rec


                # ---------------------------
                # # compute ∂λ̄_t^h / ∂φ^{sr}_m
                # # ---------------------------

            for s in range(num_comm):      # destination index of φ^{sr}_m
                for r in range(num_comm):  # source index of φ^{sr}_m
                    for m in range(1, p + 1):  # lag index (only m <= p are parameters)
                            param_idx = num_mu + (s * num_comm + r) * p + (m - 1)
                            #print(param_idx)

                            # ----- base term: δ_{hs} α_r p_{r→s} λ̄_{t+1-m}^r -----
                            if h == s and tau - m >= 0:
                                base = alpha[r] * prob_matrix[r, s] * lambda_bar[tau - m, r]
                            else:
                                base = 0.0
                
                            rec = np.dot(alpha * prob_matrix[:,h], np.sum(phi[:tau, h, :][::-1, :] * grad_lambda_bar[:tau, :, param_idx] , axis=0))
                                    # phi_time_idx = tau - k_idx - 1  # 0-based index, >= 0
                                    # rec += (
                                    #     alpha[l]
                                    #     * prob_matrix[l, h]
                                    #     * phi[phi_time_idx, h, l]
                                    #     * grad_lambda_bar[k_idx, l, param_idx]
                                    # )

                            grad_lambda_bar[tau, h, param_idx] = base + rec
            #grad_lambda_bar[tau, h, :num_mu] = 0 
    
    return lambda_bar, grad_lambda_bar

def loglik_approx(Y, comm_sizes, lambda_bar, eps=1e-12):

    Y = np.asarray(Y)
    lambda_bar = np.asarray(lambda_bar)
    comm_sizes = np.asarray(comm_sizes)

    T, q = Y.shape

    # compute Λ̄^h = average of Y_t^h over time
    Abar = Y.mean(axis=0)/comm_sizes  # shape (q,)
    Abar = np.maximum(Abar, eps)  # avoid division by zero

    # term inside parentheses
    ratio = lambda_bar / Abar  # shape (T, q)

    term = 2 * ratio - 0.5 * ratio**2  # shape (T, q)

    # final likelihood
    ll = np.sum(Y * term - comm_sizes.reshape(1, -1) * lambda_bar)

    return ll

def objective_and_grad(mu, phi, alphas, prob_matrix, comm_sizes, p, X, labs):
    """
    X    : (T, N) node-level jumps (here X_comb)
    labs : (N,) community labels
    """
    T = X.shape[0]
    num_comm = len(comm_sizes)

    # λ̄ and its gradient wrt θ
    lambda_bar, grad_lambda_bar = lambda_bar_and_grad_for_all_times(
        mu, phi, alphas, prob_matrix, comm_sizes, p, T
    )

    Y = jumps_per_community_over_time(labs, X)

    # gradient (approximate) and log-likelihood
    grad = grad_loglik_intuitive(labs, comm_sizes, X, lambda_bar, grad_lambda_bar)
    loglik = loglik_approx(Y, comm_sizes, lambda_bar)

    return loglik, grad


def gradient_descent_fit(mu0, phi0, alphas, prob_matrix, comm_sizes, p, X, labs,
                         lr_mu=1e-3, lr_phi=1e-3, max_iter=200):
    """
    Gradient ascent on the approximate log-likelihood.

    Tracks:
      - theta_history: (max_iter+1, num_params)
      - loglik_history: (max_iter+1,)
    """
    num_comm = len(comm_sizes)

    # initial parameters
    mu = mu0.astype(float).copy()
    phi = phi0.astype(float).copy()

    # flatten phi into parameter vector
    theta = np.concatenate((mu, phi_tensor_to_param_vector(phi, p)))

    mu_history = [mu.copy()]
    phi_history = [phi.copy()]
    loglik_history = []

    for it in range(max_iter):
        # current mu, phi from theta (for safety / consistency)
        mu = theta[:num_comm]
        phi = phi_param_vector_to_tensor(theta[num_comm:], num_comm, p)

        # objective and gradient at current parameters
        loglik, grad = objective_and_grad(mu, phi, alphas, prob_matrix,
                                          comm_sizes, p, X, labs)
        loglik_history.append(loglik)
        
        lr = np.concatenate([np.full(num_comm, lr_mu), np.full(num_comm*num_comm*p, lr_phi)])
        # gradient ascent step
        theta += lr * grad

        # optional: keep parameters non-negative
        theta = np.maximum(theta, 1e-11)

        mu_history.append(theta[:num_comm].copy())
        phi_history.append(phi_param_vector_to_tensor(theta[num_comm:], num_comm, p).copy())

        if (it + 1) % 20 == 0:
            print(f"iter {it+1:4d}  loglik = {loglik:.6f}")

    # final unpack
    mu_final = theta[:num_comm]
    phi_final = phi_param_vector_to_tensor(theta[num_comm:], num_comm, p)

    return mu_final, phi_final, np.array(mu_history), np.array(phi_history), np.array(loglik_history)



### Functions for APPROACH 2


