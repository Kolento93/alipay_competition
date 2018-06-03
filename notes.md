加粗的表示向量，例如：${\bf a}, {\bf x}$, 未加粗的表示标量,例如: $x,y$

规定三种导数(只是符号，计算方便而已)

1. 向量对标量求导
$$\left(\frac{\partial {\bf a}}{\partial x}\right)_i = \frac{\partial a_i}{\partial x}$$
2. 标量对向量求导
$$\left(\frac{\partial x}{\partial {\bf a}}\right)_i = \frac{\partial x}{\partial a_i}$$
2. 向量对向量求导
$$\left(\frac{\partial {\bf a}}{\partial {\bf b}}\right)_{ij} = \frac{\partial a_i}{\partial b_j}$$

其中 $A_{ij}$ 表示 $A$ 的第 $ij$ 个分量。

例1：
$$
\frac{\partial}{\partial {\bf x}}({\bf x^Ta}) = 
\frac{\partial}{\partial {\bf x}}({\bf a^Tx}) = 
{\bf a}
$$

证明：$${\bf x^Ta} = {\bf a^Tx} = \sum_{k = 1}^{n}a_kx_k$$ 是标量, 根据定义
$$\left(\frac{\partial}{\partial {\bf x}}({\bf x^Ta})\right)_i = 
\left(\frac{\partial}{\partial {\bf x}}({\bf a^Tx})\right)_i = 
\frac{\partial}{\partial x_i}({\bf x^Ta}) = a_i
$$
因此
$$
\frac{\partial}{\partial {\bf x}}({\bf x^Ta}) = 
\frac{\partial}{\partial {\bf x}}({\bf a^Tx}) = 
{\bf a}
$$


