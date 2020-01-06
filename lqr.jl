# https://www.mathworks.com/help/control/ref/lqr.html
# X = [x, xd, theta, thetad]
using LinearAlgebra

function lqr()

    Nx = 4
    Nu = 1

    # Dynamics
    Ac = [
        0.0 1.0 0.0 0.0;
        0.0 -0.1 3.0 0.0;
        0.0 0.0 0.0 1.0;
        0.0 -0.5 30.0 0.0
    ]

    Bc = [0.0; 2.0; 0.0; 5.0]

    dt = 0.1

    A = Matrix{Float64}(I, Nx, Nx) + Ac * dt
    B = Bc * dt

    # Quadratic cost matricies
    Cxx = [
        1.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0;
        0.0 0.0 1.0 0.0;
        0.0 0.0 0.0 0.0
        ]
    Cuu = reshape([1.0], 1, 1)
    Cxu = zeros(Nx, Nu)
    Cux = transpose(Cxu)
    C = [Cxx Cxu; Cux Cuu]

    # Linear cost matricies
    cx = [0.0; 0.0; 0.0; 0.0]
    cu = [0.0]
    c = [cx; cu]

    N_steps = 100

    X_initial = [1.0; 0.0; 0.0; 0.0]
    X_final = [0.0; 0.0; 0.0; 0.0]

    # Final conditions
    X_i = X_final
    K = -Cuu \ Cux
    k = -Cuu \ cu
    K_transpose = transpose(K)
    V_j = Cxx + Cxu * K + K_transpose * Cux + K_transpose * Cuu * K
    v_j = cx + Cxu * k + K_transpose * cu + K_transpose * Cuu * k # TODO: Check 3rd term

    # Memory
    K_memory = []
    k_memory = []

    # Backwards pass
    for i in N_steps:-1:1

        # Dynamics
        A_i = A
        B_i = B
        F_i = hcat(A_i, B_i)
        F_i_transpose = Array(transpose(F_i))
        f_i = zeros(Nx, Nu)

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

        # Cost matricies
        Q_i = C_i + F_i_transpose * V_j * F_i
        q_i = c_i + F_i_transpose * V_j * f_i + F_i_transpose * v_j

        Qxx_i = Q_i[1:Nx, 1:Nx]
        Qxu_i = Q_i[1:Nx, Nx+1:Nx+Nu]
        Qux_i = Q_i[Nx+1:Nx+Nu, 1:Nx]
        Quu_i = Q_i[Nx+1:Nx+Nu, Nx+1:Nx+Nu]

        qx_i = q_i[1:Nx]
        qu_i = q_i[Nx+1:Nx+Nu]

        # TODO: Calculate cost
        K_i = -Quu_i \ Qux_i
        k_i = -Quu_i \ qu_i
        K_i_transpose = transpose(K_i)

        push!(K_memory, K_i)
        push!(k_memory, k_i)

        # Value matricies
        V_i = Qxx_i + Qxu_i * K_i + K_i_transpose * Qux_i + K_i_transpose * Quu_i * K_i
        v_i = qx_i + Qxu_i * k_i + K_i_transpose * qu_i + K_i_transpose * Quu_i * k_i # TODO: Check 3rd term

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
        f_i = zeros(Nx, Nu)

        control_i = K_memory[i] * X_i + k_memory[i]

        push!(X_memory, X_i)
        push!(control_memory, control_i)

        # Step forward in time
        X_j = F_i * [X_i; control_i] + f_i
        X_i = X_j

    end

    print(X_memory)

end
