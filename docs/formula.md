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
$$ {test}
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



## Differential Dynamic Programming

If we consider that ${F_{T-1}}$ is infact a function of both $X$ and $u$ then the complexity increase. If we linearize about a set of states and control.
$$
\begin{aligned}
{F_{T}} &= f(X_{T}, u_{T}) \\\\

&\approx f(\hat{X_{T}}, \hat{u_{T}}) \\\\


{f_{T-1}} &= f(X_{T-1}, u_{T-1}) \\\\
\end{aligned}
$$

### Non Linear

Approximating non-linear model as first order
$$
\begin{aligned}

X_{T} &= f(X_{T-1}, u_{T-1}) \\\\

\begin{bmatrix}
X_{T-1} - \bar{X_{T-1}} \\
u_{T-1} - \bar{u_{T-1}}
\end{bmatrix} &= \begin{bmatrix}
\delta X_{T-1} \\
\delta u_{T-1}
\end{bmatrix} \\\\

f(X_{T-1}, u_{T-1}) &\approx 
f(\bar{X_{T-1}}, \bar{u_{T-1}}) + 
\nabla_{\bar{X_{T-1}}, \bar{u_{T-1}}} f(\bar{X_{T-1}}, \bar{u_{T-1}})  \begin{bmatrix}
\delta X_{T-1} \\
\delta u_{T-1}
\end{bmatrix} \\\\

f(\delta X_{T-1}, \delta u_{T-1}) &\approx 
\nabla_{\bar{X_{T-1}}, \bar{u_{T-1}}} f(\bar{X_{T-1}}, \bar{u_{T-1}})  \begin{bmatrix}
\delta X_{T-1} \\
\delta u_{T-1}
\end{bmatrix} \\\\

\end{aligned}
$$
As second order
$$
\begin{aligned}

X_{T} &= f(X_{T-1}, u_{T-1}) \\\\

f(X_{T-1}, u_{T-1}) &\approx 
f(\bar{X_{T-1}}, \bar{u_{T-1}}) + 
\nabla_{\bar{X_{T-1}}, \bar{u_{T-1}}} f(\bar{X_{T-1}}, \bar{u_{T-1}})  \begin{bmatrix}
\delta X_{T-1} \\
\delta u_{T-1}
\end{bmatrix}

+ \\ & \frac{1}{2} \Bigg(
\nabla^2_{\bar{X_{T-1}}, \bar{u_{T-1}}} f(\bar{X_{T-1}}, \bar{u_{T-1}}) \cdot
\begin{bmatrix}
\delta X_{T-1} \\
\delta u_{T-1}
\end{bmatrix}
\bigg) \begin{bmatrix}
\delta X_{T-1} \\
\delta u_{T-1}
\end{bmatrix} \\\\

f(\delta X_{T-1}, \delta u_{T-1}) &\approx 
\nabla_{\bar{X_{T-1}}, \bar{u_{T-1}}} f(\bar{X_{T-1}}, \bar{u_{T-1}})  \begin{bmatrix}
\delta X_{T-1} \\
\delta u_{T-1}
\end{bmatrix} + \frac{1}{2} \Bigg(
\nabla^2_{\bar{X_{T-1}}, \bar{u_{T-1}}} f(\bar{X_{T-1}}, \bar{u_{T-1}}) \cdot
\begin{bmatrix}
\delta X_{T-1} \\
\delta u_{T-1}
\end{bmatrix}
\bigg) \begin{bmatrix}
\delta X_{T-1} \\
\delta u_{T-1}
\end{bmatrix} \\\\

f(\delta X_{T-1}, \delta u_{T-1}) &\approx \Bigg(
\nabla_{\bar{X_{T-1}}, \bar{u_{T-1}}} f(\bar{X_{T-1}}, \bar{u_{T-1}}) + \frac{1}{2} 
\nabla^2_{\bar{X_{T-1}}, \bar{u_{T-1}}} f(\bar{X_{T-1}}, \bar{u_{T-1}}) \cdot
\begin{bmatrix}
\delta X_{T-1} \\
\delta u_{T-1}
\end{bmatrix}
\bigg)
\begin{bmatrix}
\delta X_{T-1} \\
\delta u_{T-1}
\end{bmatrix}  \\\\

\nabla_{\bar{X_{T-1}}, \bar{u_{T-1}}} f(\bar{X_{T-1}}, \bar{u_{T-1}}) &\rarr F_{T-1} \\\\

\nabla^2_{\bar{X_{T-1}}, \bar{u_{T-1}}} f(\bar{X_{T-1}}, \bar{u_{T-1}}) &\rarr \dot{F_{T-1}}

\end{aligned}
$$


Now:
$$
\begin{aligned}
Q(\delta X_{T-1}, \delta u_{T-1})  &= 
\frac{1}{2}
\begin{bmatrix} \delta X_{T-1} \\ \delta u_{T-1} \end{bmatrix}^T
\Bigg(
C_{T-1} + { \bigg({F_{T-1}} + \frac{1}{2} \dot{F_{T-1}} \cdot \begin{bmatrix} \delta X_{T-1} \\ \delta u_{T-1} \end{bmatrix} \bigg)}^T  \bold {V_T}  { \bigg({F_{T-1}}+ \frac{1}{2}\dot{F_{T-1}} \cdot \begin{bmatrix} \delta X_{T-1} \\ \delta u_{T-1} \end{bmatrix} \bigg)}
\Bigg)
\begin{bmatrix} \delta X_{T-1} \\ \delta u_{T-1} \end{bmatrix} + \\ &
\begin{bmatrix} \delta X_{T-1} \\ \delta u_{T-1} \end{bmatrix}^T
\Bigg(
c_{T-1} + { \bigg({F_{T-1}}+ \frac{1}{2} \dot{F_{T-1}} \cdot \begin{bmatrix} \delta X_{T-1} \\ \delta u_{T-1} \end{bmatrix} \bigg)}^T \bold {V_T} f_{T-1} +{ \bigg({F_{T-1}}+ \frac{1}{2}\dot{F_{T-1}} \cdot \begin{bmatrix} \delta X_{T-1} \\ \delta u_{T-1} \end{bmatrix} \bigg)}^T \bold{v_T}
\Bigg) + \\ &
{f_{T-1}}^T \bold{v_T}
\end{aligned}
$$


Ignoring the terms with higher than quadratic $\delta X$ and $\delta u$ terms.
$$
\begin{aligned}
Q(\delta X_{T-1}, \delta u_{T-1})  &= 
\frac{1}{2}
\begin{bmatrix} \delta X_{T-1} \\ \delta u_{T-1} \end{bmatrix}^T
\Bigg(
C_{T-1} + {F_{T-1}}^T  \bold {V_T}  F_{T-1} 
\Bigg)
\begin{bmatrix} \delta X_{T-1} \\ \delta u_{T-1} \end{bmatrix} + \\ &
\begin{bmatrix} \delta X_{T-1} \\ \delta u_{T-1} \end{bmatrix}^T
\Bigg(
c_{T-1} + { \bigg({F_{T-1}}+ \frac{1}{2} \dot{F_{T-1}} \cdot \begin{bmatrix} \delta X_{T-1} \\ \delta u_{T-1} \end{bmatrix} \bigg)}^T \bold {V_T} f_{T-1} +{ \bigg({F_{T-1}} + \frac{1}{2} \dot{F_{T-1}} \cdot \begin{bmatrix} \delta X_{T-1} \\ \delta u_{T-1} \end{bmatrix} \bigg)}^T \bold{v_T}
\Bigg) + \\ &
{f_{T-1}}^T \bold{v_T}
\end{aligned}
$$
Expanding out
$$
\begin{aligned}
Q(\delta X_{T-1}, \delta u_{T-1})  &= 
\frac{1}{2}
\begin{bmatrix} \delta X_{T-1} \\ \delta u_{T-1} \end{bmatrix}^T
\Bigg(
C_{T-1} + {F_{T-1}}^T  \bold {V_T}  F_{T-1} 
\Bigg)
\begin{bmatrix} \delta X_{T-1} \\ \delta u_{T-1} \end{bmatrix} + \\\\ &


\begin{bmatrix} \delta X_{T-1} \\ \delta u_{T-1} \end{bmatrix}^T
\Bigg(
c_{T-1} + {F_{T-1}}^T \bold {V_T} f_{T-1} + 
{\bigg(\frac{1}{2} \dot{F_{T-1}} \cdot \begin{bmatrix} \delta X_{T-1} \\ \delta u_{T-1} \end{bmatrix}\bigg)}^T  \bold {V_T} f_{T-1} +
{ {F_{T-1}}^T \bold{v_T} + 
\bigg(\frac{1}{2} \dot{F_{T-1}} \cdot \begin{bmatrix} \delta X_{T-1} \\ \delta u_{T-1} \end{bmatrix} \bigg)}^T \bold{v_T}
\Bigg)

+ \\\\ &
{f_{T-1}}^T \bold{v_T}
\end{aligned}
$$




Finding gradient with respect to control input
$$
\nabla_u Q(X_{T-1}, u_{T-1})
$$






---

$J$ - Total cost

$T$ - Transition function

$g$ - Cost of state/control pair

$\bar{x}$ - nominal trajectory



### Taylor Series Expansions

#### Generic Form

$$
\begin{aligned}

f(x) &\approx f(\bar{x}) + \frac{\partial f(x)}{\partial x} \Bigg|_{\bar{x}}\delta x + \frac{1}{2} \space  \delta x^T \space \frac{\partial^2 f(x)}{\partial x \partial x^T} \Bigg|_{\bar{x}} \delta x

\\\\

\delta x &= x-\bar{x}

\end{aligned}
$$



#### Composite Functions

Quadratic Taylor series expansion of a composite scalar function
$$
f(g(x)) \approx f(g(\bar x)) + \dot g(x) \dot f(g(x)) \delta x + \\\frac{1}{2} \bigg( 
\dot g(x)^2 \ddot f(g(x)) + \ddot g(x) \dot f(g(x))
\bigg) \delta x^2
$$
Quadratic Taylor series expansion of a multi-variable vector function
$$
f(g(x)) \approx f(g(\bar x)) + \dot g(x) \dot f(g(x)) \delta x + \\\frac{1}{2} \delta x^T \bigg( 
\dot g(x)^T \ddot f(g(x)) \dot g(x) + \ddot g(x) \dot f(g(x))
\bigg) \delta x
$$


#### Multi-Variable Vector Scalar Function Taylor Series Expansion

$$
\begin{aligned}

\frac{\partial f(x)}{\partial x} \Bigg|_{\bar{x}} &= \begin{bmatrix} 
\frac{\partial f(x)}{\partial x_1}  \\
\vdots \\
\frac{\partial f(x)}{\partial x_N}\\
\end{bmatrix}

\\\\

\frac{\partial^2 f(x)}{\partial x \partial x^T} \Bigg|_{\bar{x}}  &= \begin{bmatrix} 
\frac{\partial^2 f(x)}{\partial x_1 \partial x_1}  & \cdots & \frac{\partial^2 f(x)}{\partial x_1 \partial x_N} \\
\vdots & \ddots & \vdots\\
\frac{\partial^2 f(x)}{\partial x_N \partial x_1} & \cdots & \frac{\partial^2 f(x)}{\partial x_N \partial x_N}\\
\end{bmatrix}



\end{aligned}
$$

#### Multi-Variable Vector Valued Function Taylor Series Expansion

$$
\begin{aligned}

f(x) &= \begin{bmatrix} f_1(x) & \cdots & f_M(x)\end{bmatrix} 

\\\\

f(x) &\approx f(\bar{x}) + \frac{\partial f(x)}{\partial x} \Bigg|_{\bar{x}}\delta x + \frac{1}{2} \space  \delta x^T \space \frac{\partial^2 f(x)}{\partial x \partial x^T} \Bigg|_{\bar{x}} \delta x

\\\\

\frac{\partial f(x)}{\partial x} \Bigg|_{\bar{x}} &= \begin{bmatrix} 
\frac{\partial f_1(x)}{\partial x_1}  & \cdots & \frac{\partial f_1(x)}{\partial x_N} \\
\vdots & \ddots & \vdots\\
\frac{\partial f_M(x)}{\partial x_N} & \cdots & \frac{\partial f_M(x)}{\partial x_N}\\
\end{bmatrix}

\\\\

\frac{\partial^2 f(x)}{\partial x \partial x^T} \Bigg|_{\bar{x}}  &= Tensor \space Notation

\end{aligned}
$$



#### Quadratic Taylor Expansion of Generic functions

Define a generic function $f(x, u)$ and its quadratic approximation $\hat f(x,u)$
$$
\begin{aligned}

\hat f(x, u) &= f(\bar{x}, \bar{u}) + \delta x^T A \delta x + \delta u^T B \delta x + \delta u^T C \delta u + D^T \delta u + E \delta x

\\\\

\bigg(\frac{\partial \hat f(x, u)}{\partial u} \bigg)^T &= 2 C \delta u + B \delta u + D

\end{aligned}
$$


### Differential Dynamic Programming

#### Notes

* When we take a quadratic approximation of a function it is about a nominal state $\hat x$ and control input $\hat u$. That quadratic approximation is valid for that time step in the trajectory and is therefore denoted with a timestamp, whereas the original non-linear function is not.

#### Transition function

Define a transition function $T(x, u)$
$$
x_{t+1} = T(x_t, u_t)
$$


#### Loss function

Define a loss function $g(x, u)$ and its quadratic approximation $\hat g_t(x,u)$



#### Return function

Define a control law $u(x)$, and a return function $R(x)$  along with its quadratic approximation $\hat R_t(x)$ 
$$
\begin{aligned}

R(x_t) &= g(x_t, u(x_t))

\\\\

\hat R_t(x_t) &= \hat g(x_t, \bar{u_t} + \delta u)

\end{aligned}
$$
Now assume the control law will take the form of a linear gain on the $\delta x$ term to derive the $\delta u$ term
$$
\begin{aligned}

\delta u &= \alpha + \beta \delta x

\\\\

\hat R_t(x) &= \hat g(x, \bar u + \alpha + \beta \delta x)

\end{aligned}
$$


#### Q function

Define a Q(uality) function along with its quadratic approximation. 

Note that here, we are counting the loss associated with the current state & control combination, but the return function of the expected next state, given by the transition function.
$$
\begin{aligned}

Q(x,u) &= g(x,u) + R(T(x,u))

\\\\

\hat Q_t(x,u) &= \hat g_t(x_t,u_t) + \hat R_{t+1}(T(x,u))
\end{aligned}
$$


The Taylor series expansion of  $\hat g_t(x_t, u_t)$ is derived as follows
$$
\hat g_t(x_t, u_t) \approx g_t(\bar x_t, \bar u_t) + 
g_{x} \delta x +
g_{u} \delta u + \\
\frac{1}{2} \delta x^T g_{xx} \delta x +
\delta u^T g_{xu} \delta x +
\frac{1}{2} \delta u^T g_{uu} \delta u
$$


Using the derivation of the Taylor series expansion for a composite function, $\hat R_{t+1}(T(x,u))$ is expanded to
$$
\begin{aligned}

\hat R_{t+1}(T(x,u)) &\approx 
R(T(\bar x,\bar u)) +

\dot T(x) \dot R_{t+1}(T(x,u)) \delta x + \\ 
&\space \frac{1}{2} \delta x^T \bigg( 
\dot T(x)^T \ddot R_{t+1}(T(x,u)) \dot T(x)+ \ddot T(x) \dot R_{t+1}(T(x)) 
\bigg) \delta x

\\\\

& \approx dd
\end{aligned}
$$


#### Control modifier

Derive control modifier $\delta u$ to minimize the quadratic approximation of the Q function 
$$
\begin{aligned}

\bigg(\frac{\partial \hat g(x, u)}{\partial u} \bigg)^T &= 0

\\\\

2 C \delta u + B \delta x + D &= 0

\\\\

\delta u &= -\frac{1}{2} C^{-1} (D + B\delta x)

\\\\

\delta u &= \alpha + \beta \delta x

\end{aligned}
$$


Assign these to previous derived linear control gains
$$
\begin{aligned}

\alpha &= -\frac{1}{2} C^{-1} D

\\\\

\beta &= -\frac{1}{2} C^{-1} B

\end{aligned}
$$


