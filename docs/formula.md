## Differential Dynamic Programming

#### Transition function

Define a transition function $T(x, u)$
$$
x_{t+1} = T(x_t, u_t)
$$


#### Loss function

A cost function $g(x, u)$ designates a weighted penalty for applying control input $u$ at state $x$. Define a loss function and its quadratic approximation $\hat g_t(x,u)$
$$
\begin{aligned}

\hat g =& g +
g_x \delta x + 
g_u \delta u + 
\frac{1}{2} \bigg(
\delta x^T g_{xx} \delta x + 
\delta x^T g_{xu} \delta u + 
\delta u^T g_{ux} \delta x + 
\delta u^T g_{uu}\delta u 
\bigg)
\end{aligned}
$$


#### Return function

In a controlled dynamic system, the Return function $R(X_t)$ represents the loss of the system between the interval $[t, T]$ if it is currently at state $x$ and chooses subsequent control inputs optimally.

Define a control law $u(x)$, and a return function $R(x)$  along with its quadratic approximation $\hat R_t(x)$ 
$$
\begin{aligned}

R(x_t) &= g(x_t, u(x_t))

\\\\

\hat R_t(x_t) &= \hat g(x_t, \bar{u_t} + \delta u)

\end{aligned}
$$


Now assume the optimal control law will take the form of a linear gain on the $\delta x$ term to derive the $\delta u$ term
$$
\begin{aligned}

\delta u =& \alpha + \beta \delta x

\\\\

\hat R_t(x_t) \approx& \hat g(x_t, \bar u_t + \alpha + \beta \delta x)

\\\\

\approx& \bar g +
g_x \delta x + 
g_u (\alpha + \beta \delta x) + \\&
\frac{1}{2} \bigg(
\delta x^T g_{xx} \delta x + 
\delta x^T g_{xu} (\alpha + \beta \delta x) + 
(\alpha + \beta \delta x)^T g_{ux} \delta x + 
(\alpha + \beta \delta x)^T g_{uu} (\alpha + \beta \delta x)
\bigg)

\\\\

\approx& \bar g +
g_x \delta x + 
g_u \alpha + g_u \beta \delta x + \\&
\frac{1}{2} \bigg(
\delta x^T g_{xx} \delta x + 
\delta x^T g_{xu} \alpha + 
\delta x^T g_{xu} \beta \delta x + 
\alpha^T g_{ux} \delta x + 
(\beta \delta x)^T g_{ux} \delta x + \\&
(\beta \delta x)^T g_{uu}  \alpha + 
\alpha^T g_{uu} \beta \delta x + 
\alpha^T g_{uu} \alpha + 
(\beta \delta x)^T g_{uu} \beta \delta x
\bigg)

\\\\

\approx& \bar g +
g_x \delta x + 
g_u \alpha + g_u \beta \delta x + \\&
\frac{1}{2} \bigg(
\delta x^T g_{xx} \delta x + 
\delta x^T g_{xu} \alpha + 
\delta x^T g_{xu} \beta \delta x + 
\alpha^T g_{ux} \delta x + 
\delta x^T \beta ^T g_{ux} \delta x + \\&
\delta x^T \beta ^T g_{uu}  \alpha + 
\alpha^T g_{uu} \beta \delta x + 
\alpha^T g_{uu} \alpha + 
\delta x^T \beta ^T g_{uu} \beta \delta x
\bigg)

\\\\

\approx& \bar g + \frac{1}{2} \delta x^T R_{xx} \delta x + \delta x^T R_{x} + const

\\\\

R_{xx} =& g_{xx} + g_{xu} \beta + \beta^T g_{ux} + \beta^T g_{uu} \beta

\\\\

R_x =& g_x + g_u \beta + g_{xu} \alpha + \alpha^T g_{uu} \beta

\\\\

const =& g_u \alpha + \alpha^T g_{uu} \alpha
\end{aligned}
$$
​	

#### Q function



The Q(uality)-Function $Q(x_t,u_t)$ simiarly represents the optmal cost of the system between the interval $[t, T]$ if it is current at state $x$ **IF** it elects to use control input $u$

If we assume an optimal controller, then
$$
\min_u Q(x_t, u_t) = R(X_t)
$$
From Bellmans Equation
$$
Q(x_t, u_t) = g(x_t, u_t)+ R(x_{t+1})
$$


Note that here, we are counting the loss associated with the current state & control combination, but the return function of the expected next state. This next state can be estimated from the current state through the transition function.
$$
\begin{aligned}

Q(x_t,u_t) &= g(x_t,u_t) + R(x_{t+1})

\\\\

\hat Q_t(x_t,u_t) &= \hat g_t(x_t,u_t) + \hat R_{t+1}(T(x_t,u_t))
\end{aligned}
$$


Using the derivation of the Taylor series expansion for a composite function, the return component of the Q function - $\hat R_{t+1}(T(x_t,u_t))$ is expanded out as follows
$$
\begin{aligned}

X_{_t} =& \begin{bmatrix} x_{_t} & u_{_t} \end{bmatrix}^T

\\\\

T(\bar X_{_t}) =& \bar x_{_{t+1}}

\\\\

T_X  =& \begin{bmatrix} T_x & T_u \end{bmatrix}

\\\\

T_{XX} =& \big( 
\begin{bmatrix} T_x & T_u \end{bmatrix}_x, 
\begin{bmatrix} T_x & T_u \end{bmatrix}_u
\big)

\\\\

\hat R_{_{t+1}}(T(X_{_t})) \approx & 
R(T(\bar X_{_t})) +
R_{x_{t+1}}(T(\bar X_{_t}))^T T_X(\bar X_{_t}) \delta X + \\&
\frac{1}{2} \delta X^T \bigg( 
T_X(\bar X_{_t})^T  R_{xx_{t+1}}(T(\bar X_{_t})) T_X(\bar X_{_t}) + T_{XX}(\bar X_{_t}) R_{x_{_{t+1}}}(T(\bar X_{_t})) 
\bigg) \delta X

\\\\

\approx &
R' +
R_x'^T T_X \delta X + 
\frac{1}{2} \delta X^T \bigg( 
 T_X ^T R_{xx}' T_X  + T_{XX}  R_x' 
\bigg) \delta X

\\\\

\approx &
R' +
R_x' T_x  \delta x + 
R_x' T_u \delta u + 
\frac{1}{2} \delta X^T \bigg( 
T_X ^T R_{xx}' T_X  + T_{XX}  R_x' 
\bigg) \delta X

\\\\

\approx &
R'+ 
R_x' T_x  \delta x + 
R_x' T_u \delta u + 
\frac{1}{2} \begin{bmatrix} \delta x^T & \delta u^T \end{bmatrix} \bigg(
\begin{bmatrix}
	T_x^T R_{xx}' T_x & 
	T_x^T R_{xx}' T_u \\ 
	T_u^T R_{xx}' T_x & 
	T_u^T R_{xx}' T_u 
\end{bmatrix} + 
T_{XX}  R_x' 
\bigg) \begin{bmatrix} \delta x \\ \delta u \end{bmatrix}

\\\\

\approx &
R' + 
R_x' T_x  \delta x + 
R_x' T_u \delta u + 
\frac{1}{2} \bigg( 
\delta x^T T_x^T R_{xx}' T_x \delta x + 
\delta x^T T_x^T R_{xx}' T_u \delta u + \\&
\delta u^T T_u^T R_{xx}' T_x \delta x + 
\delta u^T T_u^T R_{xx}' T_u \delta u 
\bigg) \space + Tensors

\end{aligned}
$$


Plugging this, along with the quadratic expansion of the loss function $$\hat g_t(x_t,u_t) $$ into $$\hat Q_t(x_t,u_t)$$
$$
\begin{aligned}

\hat Q_t(x_t,u_t) \approx& \bar Q + Q_u \delta u + Q_x \delta x \frac{1}{2} \Big( \delta x^T Q_{xx} \delta x + \delta u^T Q_{ux} \delta x + \delta x^T Q_{xu} \delta u + \delta u^T Q_{uu} \delta u \Big)

\\\\
Q_{xx} =& g_{xx} + T_x^T R_{xx}'T_x
\\\\
Q_{uu} =& g_{uu} + T_u^T R_{xx}'T_u
\\\\
Q_{xu} =& g_{xu} + T_x^T R_{xx}' T_u
\\\\
Q_{ux} =& g_{ux} + T_u^T R_{xx}' T_x = Q_{xu}^T
\\\\
Q_x =& g_x + R_x' T_x
\\\\
Q_u =& R_x' T_u

\end{aligned}
$$


#### Control modifier

Derive control modifier $\delta u$ to minimize the quadratic approximation of the Q function (See Appendices: Linear Algebra)
$$
\begin{aligned}

\bigg(\frac{\partial \hat Q(x, u)}{\partial u} \bigg)^T &= 0

\\\\

Q_{ux} \delta x^T + Q_{uu} \delta u + Q_u &= 0

\\\\

Q_{uu} \delta u &= -Q_u - Q_{ux} \delta x

\\\\

\delta u &= Q_{uu}^{-1}(-Q_u -Q_{ux} \delta x)

\\\\

\alpha &= -Q_{uu}^{-1} Q_u

\\\\

\beta &= -Q_{uu}^{-1} Q_{ux}

\end{aligned}
$$


# Appendices

## Taylor Series

#### Generic Form

$$
\begin{aligned}

f(x) &\approx f(\bar{x}) + \frac{\partial f(x)}{\partial x} \Bigg|_{\bar{x}}\delta x + \frac{1}{2} \space  \delta x^T \space \frac{\partial^2 f(x)}{\partial x \partial x^T} \Bigg|_{\bar{x}} \delta x

\\\\

\delta x &= x-\bar{x}

\end{aligned}
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

&= f_x(x)

\\\\

\frac{\partial^2 f(x)}{\partial x \partial x^T} \Bigg|_{\bar{x}}  &= \begin{bmatrix} 
\frac{\partial^2 f(x)}{\partial x_1 \partial x_1}  & \cdots & \frac{\partial^2 f(x)}{\partial x_1 \partial x_N} \\
\vdots & \ddots & \vdots\\
\frac{\partial^2 f(x)}{\partial x_N \partial x_1} & \cdots & \frac{\partial^2 f(x)}{\partial x_N \partial x_N}\\
\end{bmatrix}

\\\\

&= f_{xx}(x)


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

&= f_x(x)

\\\\

\frac{\partial^2 f(x)}{\partial x \partial x^T} \Bigg|_{\bar{x}}  &= Tensor \space Notation

\\\\

& = f_{xx}(x)

\end{aligned}
$$



#### Composite Functions

Quadratic Taylor series expansion of a composite scalar function
$$
f(g(x)) \approx f(g(\bar x)) + g_x(x) f_x(g(x)) \delta x + \\\frac{1}{2} \bigg( 
g_x(x)^2 f_{xx}(g(x)) + g_{xx}(x) f_x(g(x))
\bigg) \delta x^2
$$
Quadratic Taylor series expansion of a multi-variable vector function
$$
f(g(x)) \approx f(g(\bar x)) + g_x(x) f_x(g(x)) \delta x + \\ \frac{1}{2} \delta x^T \bigg( 
g_x(x)^T f_{xx}(g(x)) g_x(x) + g_{xx}(x) f_x(g(x))
\bigg) \delta x
$$


#### Quadratic Taylor Expansion of Generic functions

Define a generic function $f(x, u)$ and its quadratic approximation $\hat f(x,u)$
$$
\begin{aligned}

\hat f(x, u) &= f(\bar{x}, \bar{u}) + \delta x^T f_{xx} \delta x + \delta u^T f_{ux} \delta x + \delta x^T f_{xu} \delta u + \delta u^T f_{uu} \delta u + f_u \delta u + f_x \delta x

\end{aligned}
$$


## Linear Algebra

#### Matrix derivatives

$$
\begin{aligned}

x_T B &\rightarrow B

\\\\

x_T x &\rightarrow 2x

\\\\

x_T B x &\rightarrow  2Bx

\end{aligned}
$$



#### Properties

$$
\begin{aligned}
(AB)^T &= B^TA^T

\end{aligned}
$$



# Notes

* When we take a quadratic approximation of a function it is about a nominal state $\hat x$ and control input $\hat u$. That quadratic approximation is valid for that time step in the trajectory and is therefore denoted with a timestamp, whereas the original non-linear function is not.