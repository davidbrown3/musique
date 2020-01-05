# https://www.mathworks.com/help/control/ref/lqr.html
# X = [x, xd, theta, thetad]

# Dynamics
A = [
    0.0 1.0 0.0 0.0;
    0.0 -0.1 3.0 0.0;
    0.0 0.0 0.0 1.0;
    0.0 -0.5 30.0 0.0
]

B = [0.0; 2.0; 0.0; 5.0]

# Quadratic cost matricies
Cxx = [
    1.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0;
    0.0 0.0 1.0 0.0;
    0.0 0.0 0.0 0.0
    ]
Cuu = reshape([1.0], 1, 1)
Cxu = zeros(4, 1)
Cux = transpose(Cxu)
C = [Cxx Cxu; Cux Cuu]

# Linear cost matricies
cx = zeros(4)
cu = [0]
c = [cx; cu]

N_steps = 100

X_initial = [0; 0; 0; 0]
X_final = [1; 0; 0; 0]

# Final conditions
X_i = X_final

K = -Cuu \ Cux
k = -Cuu \ cu
K_transpose = transpose(K)
V_j = Cxx + Cxu * K + K_transpose * Cux + K_transpose .* Cuu * K # TODO: Check outer product
v_j = cx + Cxu * k + K_transpose * Cuu + K_transpose .* Cuu * k # TODO: Check 3rd term

# Memory
K_memory = []
k_memory = []

# Backwards pass
for i in N_steps:-1:1
    
    X_i_transpose = transpose(X_i)

    # Dynamics
    A_i = A
    B_i = B
    F_i = hcat(A_i, B_i)
    F_i_transpose = transpose(F_i)
    f_i = zeros(4)

    # Quadratic cost matricies
    Cxx_i = Cxx
    Cuu_i = Cuu
    Cxu_i = Cxu
    Cux_i = Cux
    C_i = C

    # Linear cost matricies
    cx_i = cx
    cu_i = cu
    c_i = c

    #
    K_i = -Cuu_i \ Cux_i
    k_i = -Cuu_i \ cu_i
    K_i_transpose = transpose(K_i)

    push!(K_memory, K_i)
    push!(k_memory, k_i)

    # Cost matricies
    global V_j
    global v_j
    Q_i = C_i + F_i_transpose * V_j * F_i
    q_i = c_i + F_i_transpose * V_j * f_i + F_i_transpose * v_j

    # TODO: Calculate cost

    # Selecting control action
    control_i = K_i * X_i + k_i

    # Value matricies
    V_i = Cxx_i + Cxu_i * K_i + K_i_transpose * Cux_i + K_i_transpose * Cuu_i * K_i
    v_i = cx_i + Cxu_i * k_i + K_i_transpose * Cux_i + K_i_transpose * Cuu_i * k_i # TODO: Check 3rd term
    
    #
    value_i = 0.5 * X_i_transpose * V_i * X_i + X_i_transpose * V_i

    # Saving value matricies for next iteration
    V_j = V_i
    v_j = v_i

end

# Forward pass
reverse!(K_memory)
reverse!(k_memory)

X_i = X_initial

X_memory = []
control_memory = []

for i in 1:1:N_steps

    # Dynamics
    A_i = A
    B_i = B
    F_i = hcat(A_i, B_i)
    f_i = zeros(4)

    control_i = K_memory[i] * X_i + k_memory[i]
    push!(X_memory, X_i)
    push!(control_memory, control_i)

    # Step forward in time
    X_i = F_i * [X_i; control_i] + f_i

end
