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
                coeff = Y_t_h * (2.0 - lmb_t_h /Abar_h) / Abar_h - n_h
            else:
                coeff = -n_h
            grad += coeff * grad_lambda_bar[t, h, :]

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
                            if h == s and tau - m >= 0:
                                base = alpha[r] * prob_matrix[r, s] * lambda_bar[tau - m, r]
                            else:
                                base = 0.0
                
                            rec = np.dot(alpha * prob_matrix[:,h], np.sum(phi[:tau, h, :][::-1, :] * grad_lambda_bar[:tau, :, param_idx] , axis=0))

                            grad_lambda_bar[tau, h, param_idx] = base + rec
    
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
            print(f"iter {it+1:4d}  loglik = {loglik}")

    # final unpack
    mu_final = theta[:num_comm]
    phi_final = phi_param_vector_to_tensor(theta[num_comm:], num_comm, p)

    return mu_final, phi_final, np.array(mu_history), np.array(phi_history), np.array(loglik_history)

def gradient_descent_fit_mu_only(mu0, phi0, alphas, prob_matrix, comm_sizes, p, X, labs,
                                  lr_mu=1e-3, max_iter=200):
    """
    Gradient ascent on the approximate log-likelihood.
    ONLY updates mu, phi is kept fixed.
    
    Tracks:
      - mu_history: (max_iter+1, num_comm)
      - loglik_history: (max_iter,)
    """
    num_comm = len(comm_sizes)
    
    # initial parameters
    mu = mu0.astype(float).copy()
    phi = phi0.astype(float).copy()  # This stays FIXED
    
    mu_history = [mu.copy()]
    loglik_history = []
    
    for it in range(max_iter):
        # objective and gradient at current parameters
        loglik, grad = objective_and_grad(mu, phi, alphas, prob_matrix,
                                          comm_sizes, p, X, labs)
        loglik_history.append(loglik)
        
        # Extract only the gradient w.r.t. mu (first num_comm elements)
        grad_mu = grad[:num_comm]
        
        # gradient ascent step - ONLY update mu
        mu += lr_mu * grad_mu
        
        # keep parameters non-negative
        mu = np.maximum(mu, 1e-11)
        
        mu_history.append(mu.copy())
        
        if (it + 1) % 20 == 0:
            print(f"iter {it+1:4d}  loglik = {loglik:.6f}  mu = {mu}")
    
    return mu, phi, np.array(mu_history), np.array(loglik_history)

def gradient_descent_fit_phi_only(mu0, phi0, alphas, prob_matrix, comm_sizes, p, X, labs,
                                   lr_phi=1e-3, max_iter=200):
    """
    Gradient ascent on the approximate log-likelihood.
    ONLY updates phi, mu is kept fixed.
    
    Tracks:
      - phi_history: (max_iter+1, p, num_comm, num_comm)
      - loglik_history: (max_iter,)
    """
    num_comm = len(comm_sizes)
    
    # initial parameters
    mu = mu0.astype(float).copy()  # This stays FIXED
    phi = phi0.astype(float).copy()
    
    phi_history = [phi.copy()]
    loglik_history = []
    
    for it in range(max_iter):
        # objective and gradient at current parameters
        loglik, grad = objective_and_grad(mu, phi, alphas, prob_matrix,
                                          comm_sizes, p, X, labs)
        loglik_history.append(loglik)
        
        # Extract only the gradient w.r.t. phi (skip first num_comm elements)
        grad_phi = grad[num_comm:]
        
        # Convert gradient vector back to phi tensor
        grad_phi_tensor = phi_param_vector_to_tensor(grad_phi, num_comm, p)
        
        # gradient ascent step - ONLY update phi
        phi += lr_phi * grad_phi_tensor
        
        # keep parameters non-negative
        phi = np.maximum(phi, 1e-11)
        
        phi_history.append(phi.copy())
        
        if (it + 1) % 20 == 0:
            print(f"iter {it+1:4d}  loglik = {loglik:.6f}")
    
    return mu, phi, np.array(phi_history), np.array(loglik_history)


### Functions for APPROACH 2

def index_to_jm(a, p):
    a0 = a - 1
    j = a0 // p          # 0..N-1
    m = a0 % p +1       # 1..p
    return j, m

def compute_d(X, adj_matrix, p):
    T, N = X.shape
    Np = N * p
    d = np.zeros((N, Np + 1))
    d[:, 0] = T

    for a in range(1, Np + 1):
        j_a, m_a = index_to_jm(a, p)
        s = np.sum(X[:T-(m_a-1), j_a])
        d[:, a] = adj_matrix[j_a, :] * s

    return d

def compute_k(X, adj_matrix, p):
    T, N = X.shape
    Np = N * p
    k = np.zeros((N, Np + 1))

    k[:, 0] = X.sum(axis=0)

    for a in range(1, Np + 1):
        j_a, m_a = index_to_jm(a, p)
        if m_a >= T:
            continue
        Y_t = X[m_a:, :]
        Y_lag = X[:T- m_a, j_a]
        contrib = np.sum(Y_t * Y_lag[:, np.newaxis], axis=0)
        k[:, a] = adj_matrix[j_a, :] * contrib

    return k


import numpy as np

def compute_Ji_single(i, X, adj_matrix, p, js, ms, dtype=np.float64):
    X = np.asarray(X)
    adj_matrix = np.asarray(adj_matrix)
    T, N = X.shape
    Np = N * p

    # Output
    J_i = np.zeros((Np + 1, Np + 1), dtype=dtype)

    # Extract time series and adjacency column for node i
    y_i = X[:, i].astype(dtype, copy=False)        # (T,)
    col_adj = adj_matrix[:, i].astype(dtype, copy=False)  # (N,)

    # (0,0) entry
    J_i[0, 0] = y_i.sum()

    # Precompute number of usable lags (if p > T-1, larger lags give zero)
    m_max = min(p, T - 1)

    # ---- 1. Precompute B[m, j] and C[m, j, k] for m=1..m_max ----
    # B[m, j] = sum_{t=m}^{T-1} y_i(t) X_{t-m, j}
    # C[m, j, k] = sum_{t=m}^{T-1} y_i(t) X_{t-m, j} X_{t-m, k}
    B = np.zeros((p + 1, N), dtype=dtype)          # index 1..p used
    C = np.zeros((p + 1, N, N), dtype=dtype)       # index 1..p used

    for m in range(1, m_max + 1):
        # y part: y_i[m:] has length T - m
        y_tail = y_i[m:]              # (T-m,)
        # X part: X[:T-m, :] has shape (T-m, N)
        X_head = X[:T - m, :].astype(dtype, copy=False)

        # B_m(j) = sum_{t=m}^{T-1} y_i(t) X_{t-m, j}
        #        = y_tail^T @ X_head
        B[m] = y_tail @ X_head        # (N,)

        # For C_m: define Z_m[t, j] = sqrt(y_i[t+m]) * X_{t, j}
        # Then C_m = Z_m^T Z_m
        sqrt_y_tail = np.sqrt(y_tail, dtype=dtype)[:, None]   # (T-m, 1)
        Z_m = X_head * sqrt_y_tail                            # (T-m, N)
        C[m] = Z_m.T @ Z_m                                    # (N, N)

    # ---- 2. First row/column J_i[a,0], J_i[0,a] (a>=1) ----
    js1 = js[1:]      # (Np,)
    ms1 = ms[1:]      # (Np,)

    # adjacency factors adj[j_a, i] for all a
    adj_js1 = col_adj[js1]              # (Np,)

    # B_for_a[a] = B[m_a, j_a]
    B_for_a = B[ms1, js1]               # (Np,)

    J_a0 = adj_js1 * B_for_a            # (Np,)
    J_i[1:, 0] = J_a0
    J_i[0, 1:] = J_a0

    # ---- 3. Main block J_i[a,b] for a,b >= 1 ----
    # lag index matrix: M[a,b] = max(m_a, m_b)
    M_idx = np.maximum.outer(ms1, ms1)   # (Np, Np), entries in {1..p}

    # j indices matrix:
    j_a_idx = np.broadcast_to(js1[:, None], (Np, Np))  # (Np, Np)
    j_b_idx = np.broadcast_to(js1[None, :], (Np, Np))  # (Np, Np)

    # base[a,b] = C[M[a,b], j_a, j_b]
    base = C[M_idx, j_a_idx, j_b_idx]                # (Np, Np)

    # adjacency outer product: adj[j_a, i] * adj[j_b, i]
    adj_outer = np.outer(adj_js1, adj_js1)           # (Np, Np)

    J_block = adj_outer * base                       # (Np, Np)

    J_i[1:, 1:] = J_block

    return J_i


def compute_Jh_aggregated(X, adj_matrix, p, labs, comm_sizes,
                          Lambda_bar=None, dtype=np.float64):
    X = np.asarray(X)
    adj_matrix = np.asarray(adj_matrix)
    labs = np.asarray(labs)

    T, N = X.shape
    Np = N * p
    q = labs.max() + 1

    # ---- compute Λ̄^i (node-level) or use approximation ----
    if Lambda_bar is None:
        # Your community-level average intensity:
        # Abar[h] = average jumps per node in community h
        # (we assume you have this function already implemented)
        jumps_comm_t = jumps_per_community_over_time(labs, X)  # shape (T, q)
        Abar_comm = jumps_comm_t.mean(axis=0) / comm_sizes     # shape (q,)

        # Approximate node-level Λ̄^i by its community average:
        Lambda_bar = Abar_comm[labs]                           # shape (N,)

    Lambda_bar = np.asarray(Lambda_bar, dtype=dtype)
    weights = 1.0 / (Lambda_bar**2)                            # (N,)

    # ---- allocate one matrix per community ----
    Jh_list = [np.zeros((Np + 1, Np + 1), dtype=dtype) for _ in range(q)]

    # ---- precompute index -> (j, m) mapping (shared for all i) ----
    js = np.empty(Np + 1, dtype=int)
    ms = np.empty(Np + 1, dtype=int)
    js[0] = 0
    ms[0] = 0
    for a in range(1, Np + 1):
        j, m = index_to_jm(a, p)
        js[a] = j
        ms[a] = m

    # ---- main streaming loop over nodes i ----
    for i in range(N):
        J_i = compute_Ji_single(i, X, adj_matrix, p, js, ms, dtype=dtype)
        print(i)
        h = labs[i]
        Jh_list[h] += weights[i] * J_i   # accumulate weighted contribution

    return Jh_list


def build_M(comm_sizes, p):
    """
    comm_sizes: iterable of community sizes (n1, n2, ..., nq)
    p: number of lags
    returns M of shape (1 + p * sum(comm_sizes), 1 + p * len(comm_sizes))
    """
    comm_sizes = np.asarray(comm_sizes)
    q = len(comm_sizes)
    N = comm_sizes.sum()

    rows = 1 + N * p
    cols = 1 + q * p
    M = np.zeros((rows, cols))

    M[0, 0] = 1.0

    row_offset = 1
    for h, n_h in enumerate(comm_sizes):
        block_rows = n_h * p
        block_cols = p
        col_offset = 1 + h * p

        block = np.vstack([np.eye(p)] * n_h) # shape (n_h*p, p)
        print(block.shape)
        print(block)
        #sns.heatmap(block)
        M[row_offset:row_offset + block_rows,
          col_offset:col_offset + block_cols] = block

        row_offset += block_rows

    return M


def compute_J(X, adj_matrix, p):
    T, N = X.shape
    Np = N * p
    J = np.zeros((N, Np + 1, Np + 1))

    for i in range(N):
        y_i = X[:, i]
        J[i, 0, 0] = y_i.sum()
        
        for a in range(1, Np + 1):
            for b in range(1, Np + 1):
                j_a, m_a = index_to_jm(a, p)
                j_b, m_b = index_to_jm(b, p)
                J[i,a, 0] = adj_matrix[j_a, i] * np.sum(y_i[m_a:] * X[:T - m_a, j_a])
                J[i,0, b] = adj_matrix[j_b, i] * np.sum(y_i[m_b:] * X[:T - m_b, j_b])
                J[i, a, b] = adj_matrix[j_a, i] * adj_matrix[j_b, i] * np.sum(y_i[max(m_a, m_b):] * X[:T - max(m_a, m_b), j_a] * X[:T - max(m_a, m_b), j_b])
    return J


def compute_J_ultra_optimized(X, adj_matrix, p):
    """
    Ultra-optimized version with maximum vectorization.
    
    Further optimizations:
    - Fully vectorized C computation
    - Vectorized main block computation using einsum
    - Minimal Python loops
    """
    T, N = X.shape
    Np = N * p
    J = np.zeros((N, Np + 1, Np + 1))
    
    # J[i, 0, 0] for all i
    J[:, 0, 0] = X.sum(axis=0)
    
    # Precompute mappings
    a_range = np.arange(1, Np + 1)
    js = (a_range - 1) // p
    ms = (a_range - 1) % p + 1
    
    # Precompute B[i, m, j] vectorized
    B = np.zeros((N, p + 1, N))
    for m in range(1, min(p + 1, T)):
        y_tail = X[m:, :].T      # (N, T-m)
        X_head = X[:T-m, :]      # (T-m, N)
        B[:, m, :] = y_tail @ X_head
    
    # Vectorized computation of J[:, a, 0] and J[:, 0, a]
    for a in range(Np):
        a_idx = a + 1
        J[:, a_idx, 0] = adj_matrix[js[a], :] * B[:, ms[a], js[a]]
        J[:, 0, a_idx] = J[:, a_idx, 0]
    
    # Precompute all C matrices at once
    m_max_unique = np.unique(np.maximum.outer(ms, ms))
    C_dict = {}
    
    for m in m_max_unique:
        if m >= T:
            continue
        y_tail = X[m:, :].T      # (N, T-m)
        X_head = X[:T-m, :]      # (T-m, N)
        
        # Vectorized: for all i simultaneously
        # C[i, m] shape: (N, N, N) where first index is node i
        weighted_X = X_head[np.newaxis, :, :] * y_tail[:, :, np.newaxis]  # (N, T-m, N)
        C_m = np.einsum('itj,itk->ijk', weighted_X, X_head[np.newaxis, :, :])  # (N, N, N)
        C_dict[m] = C_m
    
    # Vectorized main block computation
    m_max_matrix = np.maximum.outer(ms, ms)  # (Np, Np)
    
    for a in range(Np):
        print(a)
        j_a = js[a]
        a_idx = a + 1
        
        for b in range(Np):
            j_b = js[b]
            b_idx = b + 1
            m_max = m_max_matrix[a, b]
            
            if m_max >= T or m_max not in C_dict:
                continue
            
            # Vectorized across all nodes i
            adj_product = adj_matrix[j_a, :] * adj_matrix[j_b, :]
            J[:, a_idx, b_idx] = adj_product * C_dict[m_max][:, j_a, j_b]
    
    return J


def compute_nu_given_inputs(d, k, J, M, labs, X, comm_sizes):
    num_comm = len(comm_sizes)
    #num_params = num_comm * num_comm**2 * p
    nu_list = []
    start = 0
    A_bar = jumps_per_community_over_time(labs, X).mean(axis=0)/comm_sizes
    for h in range(num_comm):
        size = comm_sizes[h]
        g_h = M.T @ np.sum(d[start: start+size,:], axis=0)
        s_h = M.T @ np.sum(k[start:start+size, :], axis = 0)/A_bar[h]
        J_h = M.T @ np.sum(J[start:start+size, :,:], axis = 0)/A_bar[h]**2 @ M

        nu_h = np.linalg.inv(J_h) @ (g_h - 2 * s_h)
        nu_list.append(nu_h)

    return nu_list

def compute_nu_given_inputes_vectorized(d, k, J, M, labs, X, comm_sizes):
    """
    Fully vectorized version - computes all communities at once.
    """
    num_comm = len(comm_sizes)
    N = d.shape[0]
    
    # Compute Lambda_bar and weights
    Y_comm = jumps_per_community_over_time(labs, X)
    A_bar = Y_comm.mean(axis=0) / comm_sizes
    Lambda_bar = A_bar[labs]
    weights = 1.0 / (Lambda_bar ** 2)
    
    # One-hot encoding for community membership
    one_hot = np.eye(num_comm)[labs]  # (N, num_comm)
    
    # Vectorized g_h computation for all communities
    # g_h[h] = M.T @ sum(d[i] for i in community h)
    d_by_comm = one_hot.T @ d  # (num_comm, Np+1)
    g_all = (M.T @ d_by_comm.T).T  # (num_comm, qp+1)
    
    # Vectorized s_h computation
    weighted_k = k / Lambda_bar[:, np.newaxis]  # (N, Np+1)
    k_by_comm = one_hot.T @ weighted_k  # (num_comm, Np+1)
    s_all = (M.T @ k_by_comm.T).T  # (num_comm, qp+1)
    
    # Vectorized J_h computation (this is the expensive part)
    nu_list = []
    for h in range(num_comm):
        # Get nodes in community h
        mask = (np.array(labs) == h)
        weights_h = weights[mask]  # Shape: (size_h,)
        J_h_nodes = J[mask]        # Shape: (size_h, Np+1, Np+1)

    # Then broadcast correctly
        weighted_J = np.sum(J_h_nodes * weights_h[:, np.newaxis, np.newaxis], axis=0)
        J_h = M.T @ weighted_J @ M
        
        # Compute nu_h
        nu_h = np.linalg.inv(J_h) @ (g_all[h] - 2 * s_all[h])
        nu_list.append(nu_h)
    
