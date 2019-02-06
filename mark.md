[toc]

## SVM
### 函数间隔
对于给定训练数据集T合超平面(w,b)，定义超平面(w,b)关于样本点($x_{i}$,$y_{i}$)的函数间隔为

$$\begin{array}{c}
\hat{\gamma}_{i}=y_{i}(w \cdot x_{i} + b) 
\end{array}$$
但是，如果我们成比例变化w和b，那么间隔就会发生变化，显然不合预期，我们需要规范化w，得到几何间隔$$\begin{array}{c}
\hat{\gamma}_{i}=\frac{y_{i}}{\lVert \mathbf{w} \rVert}(w \cdot x_{i} + b)
\end{array}$$

SVM本质就是求间隔最大化的问题，让离超平面最近的点尽可能地远

### SVM对偶算法
最终SVM其实就是要想办法求得对偶问题
$$\begin{array}{c}
L(w,b,a) = \frac{1}{2}\lVert \mathbf{w} \rVert - \sum_{i=1}^{N}a_{i}(y{i}(w\cdot x_{i} + b) - 1)
\end{array}$$
令上式分别对w，b求导，并让式子等于0
$$\begin{array}{c}
\nabla L_{w} = w - \sum_{i = 1}^{N}a_{i}y_{i}x_{i} = 0 \\ \\
\nabla L_{b} = \sum_{i = 1}^{N}a_{i}y_{i} = 0
\end{array}$$
可以得到
$$\begin{array}{c}
w = \sum_{i = 1}^{N}a_{i}y_{i}x_{i} \\ \\
\sum_{i = 1}^{N}a_{i}y_{i} = 0
\end{array}$$
即我们最后要求的东西是
$$\begin{array}{c}
min_{w,b}L(w,b,a) = \frac{1}{2}\sum_{i = 1}^{N}\sum_{j = 1}^{N}a_{i}a_{j}y_{i}y_{j}(x_{i} \cdot x_{j}) + \sum_{i = 1}^{N}a_{i}
\end{array}$$
但是显然如果直接对上式进行求解，显然这个二次规划的计算量会很大，所以我们转而求它的对偶问题
$$\begin{array}{c}
max_{a} -\frac{1}{2}\sum_{i = 1}^{N}\sum_{j = 1}^{N}a_{i}a_{j}y_{i}y_{j}(x_{i} \cdot x_{j}) + \sum_{i = 1}^{N}a_{i} \\ \\
s.t \sum_{i = 1}^{N}a_{i} = 0 \\ \\
a_{i}\geq 0, \quad i = 1,2...N
\end{array}$$

再转换求最小，换个符号就好了
$$\begin{array}{c}
min_{a}\frac{1}{2}\sum_{i = 1}^{N}\sum_{j = 1}^{N}a_{i}a_{j}y_{i}y_{j}(x_{i} \cdot x_{j}) - \sum_{i = 1}^{N}a_{i} \\ \\
s.t \sum_{i = 1}^{N}a_{i} = 0 \\ \\
a_{i}\geq 0, \quad i = 1,2...N
\end{array}$$