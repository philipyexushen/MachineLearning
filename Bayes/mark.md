##Naive Bayes Algorithm
其实这玩意很简单，就是算概率
1. 先算先验概率和条件概率
$$\begin{array}{c}
P(Y = c_{k}) = \frac{\sum_{i=1}^{N}I(y_{i} = c_{k})}{N}, k=1,2,...,K \\ \\
P(X^{(j)} = a_{jl}|Y = c_{k}) = \frac{\sum_{i=1}^{N}I(x^{j} = a_{jl}, y_{i} = c_{k})}{\sum_{i=1}^{N}I(y_{i} = c_{k})} j = 1,2,...,n; l = 1,2,..., S_{j}; k = 1,2,...,K
\end{array}$$
2. 对于给定的实例$x=(x^{(1)},x^{(2)},...,x^{(n)})^{T}$，计算
$$\begin{array}{c}
P(Y = c_{k})\prod_{j=1}^nP(X^{(j)} = x^{j}|Y = c_{k}), k=1,2,...,K
\end{array}$$
3. 确定实例x的类
$$\begin{array}{c}
y = argmax_{ck}P(Y = c_{k})\prod_{j=1}^nP(X^{(j)} = x^{(j)} | Y = c_{k})
\end{array}$$

注意，朴素贝叶斯法和贝叶斯估计是两个不同的概念
所谓的条件概率的贝叶斯估计，实际上就是在原有条件概率的基础上加上拉普拉斯平滑处理
$$\begin{array}{c}
P(X^{(j)} = a_{jl}|Y = c_{k}) \frac{\sum_{i=1}^{N}I(x^{j} = a_{jl}, y_{i} = c_{k}) + \lambda}{\sum_{i=1}^{N}I(y_{i} = c_{k}) + S_{j}\lambda}
\end{array}$$
上式当$\lambda$=0时，即为极大近似估计