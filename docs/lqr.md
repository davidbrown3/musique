## Introduction

A cost function $C(X, u)$ designates a weighted penalty for applying control input $u$ at state $X$ 

In a controlled dynamic system, the Value-Function $V(X)$ represents the cost of the system between the interval $[t, T]$ if it is currently at state $X$ and chooses subsequent control inputs optimally

The Q-Function $Q(X,u)$ simiarly represents the optmal cost of the system between the interval $[t, T]$ if it is current at state $X$ **IF** it elects to use control input $u$

If we assume an optimal controller, then
$$
\min_u Q(X, u) = V(X)
$$


Bellmans Equation
$$
Q(X_T, u_T) = C(X_T, u_T)+ V(X_{T+1})
$$
Defining a quadratic cost function
$$
\begin{aligned}

C_T &= \begin{bmatrix}
C_{xx} & C_{xu} \\
C_{ux} & C_{uu}
\end{bmatrix} \\\\

c_T &= \begin{bmatrix}
C_{x} \\
C_{u}
\end{bmatrix} \\\\

c(x_T, u_T) &= \frac{1}{2}
\begin{bmatrix}
X_T \\
u_T
\end{bmatrix}^T 
C_T
\begin{bmatrix}
X_T \\
u_T
\end{bmatrix}
+ 
\begin{bmatrix}
X_T \\
u_T
\end{bmatrix}^T
c_T
\end{aligned}
$$
Populating Q function
$$
\begin{aligned}
Q(X_T, u_T) &= 
\frac{1}{2}
\begin{bmatrix}
X_T \\
u_T
\end{bmatrix}^T
C_T
\begin{bmatrix}
X_T \\
u_T
\end{bmatrix} +
\begin{bmatrix}
X_T \\
u_T
\end{bmatrix}^T
c_T + 
V(X_{T+1}) \\\\

&= 
\frac{1}{2}
\begin{bmatrix}
X_T \\
u_T
\end{bmatrix}^T
\begin{bmatrix}
C_{xx} & C_{xu} \\
C_{ux} & C_{uu}
\end{bmatrix}
\begin{bmatrix}
X_T \\
u_T
\end{bmatrix} + 
\begin{bmatrix}
X_T \\
u_T
\end{bmatrix}^T 
\begin{bmatrix}
c_x \\
c_u
\end{bmatrix} + 
V(X_{T+1})

\end{aligned}
$$
Expanding Q function
$$
Q(X_T, u_T) = 
\frac{1}{2} {X_T}^T  C_{xx}  X_T + 
\frac{1}{2} {X_T}^T C_{xu} u_T + 
\frac{1}{2} {u_T}^T C_{ux} X_T + 
\frac{1}{2} {u_T}^T C_{uu} u_T  + \\
{X_T}^T c_x + {u_T}^T c_u + V(X_{T+1})
$$
Collecting similar terms
$$
Q(X_T, u_T) = 
\frac{1}{2} {X_T}^T  C_{xx}  X_T + 
{u_T}^T C_{ux} X_T + 
\frac{1}{2} {u_T}^T C_{uu} u_T  +  \\
{X_T}^T c_x + {u_T}^T c_u + V(X_{T+1})
$$


## SPECIAL CASE WHEN AT TERMINAL TIME STEP (T=N)

Assumes quadratic value function; so we can predict how value changes with X
$$
\begin{aligned}
T &= N \\\\
V(X_{N+1}) &= 0 \\\\
\end{aligned}
$$
Finding gradient of Q function wrt. control input: [See matrix derivative cheat sheet](http://www.gatsby.ucl.ac.uk/teaching/courses/sntn/sntn-2017/resources/Matrix_derivatives_cribsheet.pdf)
$$
\nabla_u Q(X_T, u_T) = C_{ux} X_T + C_{uu} u_T + {u_T}^T c_u
$$
Assign to zero to find minima
$$
\begin{aligned}
C_{ux} X_T + C_{uu} u_T + c_u &= 0\\\\
C_{uu} u_T &= -c_u -C_{ux} X_T \\\\
u_T &= {C_{uu}}^{-1} \big( -c_u -C_{ux} X_T \big)
\end{aligned}
$$
Assigning to control gains
$$
\begin{aligned}
u_T &= K_T X_T + k_T \\\\
K_T &= -{C_{uu}}^{-1} C_{ux} \\\\
k_T &= -{C_{uu}}^{-1} c_u
\end{aligned}
$$
Plug in expression for control input into Q function. There is now no dependency on control input and it is equivalent to the value function.
$$
V(X_T) = \frac{1}{2}
\begin{bmatrix}
X_T \\
K_T X_T + k_T
\end{bmatrix}^T
\begin{bmatrix}
C_{xx} & C_{xu} \\
C_{ux} & C_{uu}
\end{bmatrix}
\begin{bmatrix}
X_T \\
K_T X_T + k_T
\end{bmatrix} + 
\begin{bmatrix}
X_T \\
K_T X_T + k_T
\end{bmatrix}^T 
\begin{bmatrix}
c_x \\
c_u
\end{bmatrix}
$$
Expand linear algebra (1)
$$
V(X_T) = \frac{1}{2} \Big[ {X_T}^T  C_{xx}  X_T + \\
{X_T}^T C_{xu} \big(K_T X_T + k_T \big) + \\
{\big(K_T X_T + k_T \big)}^T C_{ux} X_T + \\
{\big(K_T X_T + k_T \big)}^T C_{uu} \big(K_T X_T + k_T \big) \Big]  + \\
{X_T}^T c_x + \\
{\big(K_T X_T + k_T \big)}^T c_u
$$
Expand transpose expressions (2)
$$
V(X_T) = \frac{1}{2} \Big[{X_T}^T  C_{xx}  X_T + \\
{X_T}^T C_{xu} \big(K_T X_T + k_T \big) + \\
{\big({X_T}^T {K_T}^T + {k_T}^T \big)} C_{ux} X_T + \\
{\big({X_T}^T {K_T}^T + {k_T}^T \big)} C_{uu} \big(K_T X_T + k_T \big) \Big] + \\
{X_T}^T c_x + \\
{\big({X_T}^T {K_T}^T + {k_T}^T \big)} c_u
$$
Multiply out (3)
$$
V(X_T) = \frac{1}{2} \Big[{X_T}^T  C_{xx}  X_T + \\
{X_T}^T C_{xu} K_T X_T + {X_T}^T C_{xu} k_T + \\
{X_T}^T {K_T}^T C_{ux} X_T + {k_T}^T C_{ux} X_T + \\
{X_T}^T {K_T}^T C_{uu} K_T X_T + {X_T}^T {K_T}^T C_{uu} k_T + \\
{k_T}^T C_{uu} K_T X_T + {k_T}^T C_{uu} k_T \Big] + \\
{X_T}^T c_x + \\
{X_T}^T {K_T}^T c_u + {k_T}^T c_u
$$
Group (4)
$$
V(X_T) = {X_T}^T C_{xu} k_T + \\
{X_T}^T {K_T}^T C_{uu} k_T + \\
\frac{1}{2} \Big[{X_T}^T  C_{xx}  X_T + \\
{X_T}^T C_{xu} K_T X_T + \\
{X_T}^T {K_T}^T C_{ux} X_T + \\
{X_T}^T {K_T}^T C_{uu} K_T X_T + \\
{k_T}^T C_{uu} k_T \Big] + \\
{X_T}^T c_x + \\
{X_T}^T {K_T}^T c_u + {k_T}^T c_u
$$
Simplify
$$
\begin{aligned}
V(X_T) &= \frac{1}{2} {X_T}^T  \bold {V_T}  X_T + {X_T}^T \bold{v_T}\\\\

\bold {V_T} &= C_{xx} + 
C_{xu} K_T + 
{K_T}^T C_{ux} + 
{K_T}^T C_{uu} K_T \\\\

\bold{v_T} &= C_{xu} k_T + 
{K_T}^T C_{uu} k_T +
{K_T}^T c_u +
c_x \\\\

cont &= {k_T}^T C_{uu} k_T + {k_T}^T c_u

\end{aligned}
$$

Using the dynamics of the system

### Linear

$$
X_T = F_{T-1} \begin{bmatrix}
X_{T-1} \\
u_{T-1}
\end{bmatrix} + f_{T-1}

$$

Plugging into value expression
$$
V(X_{T}) = \frac{1}{2} 
\Big({F_{T-1} \begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix} + f_{T-1} \Big)}^T  \bold {V_T} 
\Big({F_{T-1} \begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix} + f_{T-1} \Big)} + \\
\Big({F_{T-1} \begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix} + f_{T-1} \Big)}^T \bold{v_T}\\\\
$$
Expanding
$$
V(X_T) = \frac{1}{2} \Bigg[
{\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix}}^T {F_{T-1}}^T  \bold {V_{T}}  F_{T-1} \begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix} +
{\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix}}^T {F_{T-1}}^T \bold {V_T} f_{T-1} + \\
{f_{T-1}}^T \bold {V_T} F_{T-1} \begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix} + 
{f_{T-1}}^T \bold {V_T} {f_{T-1}}
\Bigg] + \\
\Big({F_T \begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix}
+ f_{T-1} \Big)}^T \bold{v_T} 
$$
Grouping
$$
\begin{aligned}
V(X_T) &= {\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix}}^T {F_{T-1}}^T \bold {V_T} f_{T-1} + \\
& \frac{1}{2}
{\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix}}^T {F_{T-1}}^T  \bold {V_T}  F_{T-1} \begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix} + \\
& \Big({\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix}}^T {F_{T-1}}^T + {f_{T-1}}^T \Big) \bold{v_T}
\end{aligned}
$$

# Loop

Looping backeards through $T = [N:-1:0]$
$$
\begin{aligned}
Q(X_{T-1}, u_{T-1}) &= 
\frac{1}{2}
\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix}^T
C_{T-1}
\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix} +
\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix}^T
c_{T-1} + 
V(X_T)

\\\\\\

&= 
\frac{1}{2}
\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix}^T
C_{T-1}
\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix} +
\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix}^T
c_{T-1} + \\
& {\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix}}^T 
{F_{T-1}}^T \bold {V_T} f_{T-1} + \\
& \frac{1}{2}
{\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix}}^T 
{F_{T-1}}^T  \bold {V_T}  F_{T-1} 
\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix} + \\
& \Big(
{\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix}}^T 
{F_{T-1}}^T + {f_{T-1}}^T 
\Big) 
\bold{v_T}

\\\\\\
	
&= 
\frac{1}{2}
\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix}^T
\Bigg(
C_{T-1} + {F_{T-1}}^T  \bold {V_T}  F_{T-1} 
\Bigg)
\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix} + \\ &
\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix}^T
\Big(
c_{T-1} + {F_{T-1}}^T \bold {V_T} f_{T-1} + {F_{T-1}}^T \bold{v_T}
\Big) + \\ &
{f_{T-1}}^T \bold{v_T}
\end{aligned}
$$

Collecting terms
$$
\begin{aligned}

Q(X_{T-1}, u_{T-1}) &= 
\frac{1}{2} 
\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix}^T
\bold{Q_{T-1}}
\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix} +
\begin{bmatrix} X_{T-1} \\ u_{T-1} \end{bmatrix}^T
\bold{q_{T-1}}

\\\\

\bold{Q_{T-1}} &= C_{T-1} + {F_{T-1}}^T  \bold {V_T}  F_{T-1} 

\\\\

&= \begin{bmatrix} 
\bold{Q_{(x_{T-1}, x_{T-1})}} & \bold{Q_{(x_{T-1}, u_{T-1})}} \\
\bold{Q_{(u_{T-1}, x_{T-1})}} & \bold{Q_{(u_{T-1}, u_{T-1})}}
\end{bmatrix}

\\\\

\bold{q_{T-1}} &= c_{T-1} + {F_{T-1}}^T \bold {V_T} f_{T-1} + {F_{T-1}}^T \bold{v_T}

\\\\

&= \begin{bmatrix} \bold{q_{(x_{T-1})}} \\ \bold{q_{(u_{T-1})}} \end{bmatrix}

\end{aligned}
$$
Using same re-arranging as previously, finding gradient of Q function wrt. control input. This assumes that ${F_{T-1}}$ is constant with $X$ and $u$
$$
\begin{aligned}
u_{T-1} &= K_{T-1} X_{T-1} + k_{T-1} \\\\
K_{T-1} &= -\bold{Q_{(u_{T-1}, u_{T-1})}}^{-1} \bold{Q_{(u_{T-1}, x_{T-1})}} \\\\
k_{T-1} &= -\bold{Q_{(u_{T-1}, u_{T-1})}}^{-1} \bold{q_{(u_{T-1})}}
\end{aligned}
$$
Plug in expression for control input into Q function. Since there is no dependency on control input, this is equivalent to a value function
$$
V(X_{T-1}) = \frac{1}{2}
\begin{bmatrix}
X_{T-1} \\
K_{T-1} X_{T-1} + k_{T-1}
\end{bmatrix}^T
\begin{bmatrix} 
\bold{Q_{(x_{T-1}, x_{T-1})}} & \bold{Q_{(x_{T-1}, u_{T-1})}} \\
\bold{Q_{(u_{T-1}, x_{T-1})}} & \bold{Q_{(u_{T-1}, u_{T-1})}}
\end{bmatrix}
\begin{bmatrix}
X_{T-1} \\
K_{T-1} X_{T-1} + k_{T-1}
\end{bmatrix} + \\
\begin{bmatrix}
X_{T-1} \\
K_{T-1} X_{T-1} + k_{T-1}
\end{bmatrix}^T 
\begin{bmatrix} \bold{q_{(x_{T-1})}} \\ \bold{q_{(u_{T-1})}} \end{bmatrix}
$$

$$
\begin{aligned}
V(X_{T-1}) &= \frac{1}{2} {X_{T-1}}^T  \bold {V_{T-1}}  X_{T-1} + {X_{T-1}}^T \bold{v_{T-1}} + V(X_T)\\\\

\bold {V_{T-1}} &= \bold{Q_{(x_{T-1}, x_{T-1})}} + 
\bold{Q_{(x_{T-1}, u_{T-1})}} K_{T-1} + 
{K_{T-1}}^T \bold{Q_{(u_{T-1}, x_{T-1})}} + 
{K_{T-1}}^T \bold{Q_{(u_{T-1}, u_{T-1})}} K_{T-1} \\\\

\bold{v_{T-1}} &= \bold{Q_{(x_{T-1}, u_{T-1})}} k_{T-1} + 
{K_{T-1}}^T \bold{Q_{(u_{T-1}, u_{T-1})}} k_{T-1} +
{K_{T-1}}^T \bold{q_{(u_{T-1})}} +
\bold{q_{(x_{T-1})}} \\\\

cont &= {k_{T-1}}^T C_{uu} k_{T-1} + {k_{T-1}}^T c_u

\end{aligned}
$$

