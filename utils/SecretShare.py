"""
秘密共享方案的实现
使用numpy的ndarray作为基础操作对象
包括加法秘密共享、向量/矩阵拆分、Beaver三元组等功能
"""

import random
import numpy as np


class SecretShare:
    def __init__(self, value):
        """
        初始化一个秘密共享对象
        :param value: 可以是标量、向量或矩阵形式的值，将被转换为numpy.ndarray
        """
        if isinstance(value, np.ndarray):
            self.value = value
        elif isinstance(value, (list, tuple)):
            self.value = np.array(value, dtype=float)
        else:
            self.value = np.array(value, dtype=float)

    @classmethod
    def from_secret_shares(cls, secret_shares):
        """
        从秘密共享列表创建SecretShare对象列表
        :param secret_shares: 秘密共享生成的值列表
        :return: SecretShare对象列表
        """
        return [cls(share) for share in secret_shares]

    def __add__(self, other):
        """
        加法运算符重载
        :param other: 另一个SecretShare对象
        :return: 两个对象相加的结果
        """
        if isinstance(other, SecretShare):
            return SecretShare(self.value + other.value)
        return SecretShare(self.value + other)

    def __sub__(self, other):
        """
        减法运算符重载
        :param other: 另一个SecretShare对象
        :return: 两个对象相减的结果
        """
        if isinstance(other, SecretShare):
            return SecretShare(self.value - other.value)
        return SecretShare(self.value - other)

    def __mul__(self, other):
        """
        乘法运算符重载（点乘或矩阵乘法）
        :param other: 另一个SecretShare对象
        :return: 两个对象相乘的结果
        """
        if isinstance(other, SecretShare):
            # 如果是矩阵乘法
            if len(self.value.shape) > 1 and len(other.value.shape) > 1:
                return SecretShare(np.dot(self.value, other.value))
            # 否则是元素乘法
            return SecretShare(self.value * other.value)
        return SecretShare(self.value * other)

    def __matmul__(self, other):
        """
        矩阵乘法运算符重载 (@)
        :param other: 另一个SecretShare对象
        :return: 两个对象矩阵乘法的结果
        """
        if isinstance(other, SecretShare):
            return SecretShare(np.matmul(self.value, other.value))
        return SecretShare(np.matmul(self.value, other))

    def __str__(self):
        """
        字符串表示
        :return: 对象的字符串表示
        """
        return f"SecretShare({self.value})"

    def __repr__(self):
        """
        正式字符串表示
        :return: 对象的正式字符串表示
        """
        return self.__str__()

    @staticmethod
    def secret_share_scalar(secret, n, rand_range=None):
        """
        将标量秘密分成n个部分
        :param secret: 要分享的秘密（标量）
        :param n: 参与方的数量
        :param rand_range: 随机数生成范围（元组）
        :return: 包含n个部分的numpy数组
        """
        if rand_range is None:
            rand_range = (0, secret) if secret > 0 else (secret, 0)
        
        shares = np.random.uniform(rand_range[0], rand_range[1], n-1)
        shares = np.append(shares, secret - np.sum(shares))
        return shares

    @staticmethod
    def reconstruct_scalar(shares):
        """
        通过标量共享部分还原秘密
        :param shares: 包含n个部分的numpy数组
        :return: 还原的秘密（标量）
        """
        return np.sum(shares)

    @staticmethod
    def secret_share_vector(secret_vector, n, rand_range=None):
        """
        将秘密向量分成n个部分
        :param secret_vector: 要分享的秘密向量（numpy数组）
        :param n: 参与方的数量
        :param rand_range: 随机数生成范围（元组）
        :return: 包含n个部分的列表，每个部分也是一个numpy数组
        """
        if isinstance(secret_vector, list):
            secret_vector = np.array(secret_vector, dtype=float)

        if rand_range is None:
            rand_range = (np.min(secret_vector), np.max(secret_vector))

        dim = secret_vector.shape[0]
        shares = []

        for i in range(n):
            if i < n - 1:
                share = np.random.uniform(rand_range[0], rand_range[1], dim)
                shares.append(share)
            else:
                # 最后一个份额使总和等于原始向量
                share = secret_vector - np.sum(shares, axis=0)
                shares.append(share)

        return shares

    @staticmethod
    def reconstruct_vector(shares):
        """
        通过向量共享部分还原秘密向量
        :param shares: 包含n个部分的列表，每个部分也是一个numpy数组
        :return: 还原的秘密向量（numpy数组）
        """
        return np.sum(shares, axis=0)

    @staticmethod
    def secret_share_matrix(secret_matrix, n, rand_range=None):
        """
        将秘密矩阵分成n个部分
        :param secret_matrix: 要分享的秘密矩阵（numpy数组）
        :param n: 参与方的数量
        :param rand_range: 随机数生成范围（元组）
        :return: 包含n个部分的列表，每个部分也是一个numpy数组
        """
        if isinstance(secret_matrix, list):
            secret_matrix = np.array(secret_matrix, dtype=float)

        if rand_range is None:
            rand_range = (np.min(secret_matrix), np.max(secret_matrix))

        rows, cols = secret_matrix.shape
        shares = []

        for i in range(n):
            if i < n - 1:
                share = np.random.uniform(rand_range[0], rand_range[1], (rows, cols))
                shares.append(share)
            else:
                # 最后一个份额使总和等于原始矩阵
                share = secret_matrix - np.sum(shares, axis=0)
                shares.append(share)

        return shares

    @staticmethod
    def reconstruct_matrix(shares):
        """
        通过矩阵共享部分还原秘密矩阵
        :param shares: 包含n个部分的列表，每个部分也是一个numpy数组
        :return: 还原的秘密矩阵（numpy数组）
        """
        return np.sum(shares, axis=0)

    @staticmethod
    def generate_beaver_triplet(dim1, dim2, dim3, rand_range=None):
        """
        生成Beaver三元组：随机矩阵A、B和C，其中C = A @ B
        :param dim1: 矩阵A的行数
        :param dim2: 矩阵A的列数和矩阵B的行数
        :param dim3: 矩阵B的列数
        :param rand_range: 随机数生成范围（元组）
        :return: 矩阵A、B和C
        """
        if rand_range is None:
            rand_range = (0, 10)

        A = np.random.uniform(rand_range[0], rand_range[1], (dim1, dim2))
        B = np.random.uniform(rand_range[0], rand_range[1], (dim2, dim3))
        C = np.matmul(A, B)
        
        return A, B, C
    
    @staticmethod
    def beaver_multiply(A_shares, B_shares, beaver_triplet_shares):
        """
        使用Beaver三元组进行矩阵乘法
        :param A_shares: 矩阵A的共享部分列表
        :param B_shares: 矩阵B的共享部分列表
        :param beaver_triplet_shares: Beaver三元组的共享部分列表，每个元素是(a_share, b_share, c_share)
        :return: 矩阵C的共享部分列表，其中C = A @ B
        """
        n = len(A_shares)
        P_shares = [triplet[0] for triplet in beaver_triplet_shares]
        Q_shares = [triplet[1] for triplet in beaver_triplet_shares]
        O_shares = [triplet[2] for triplet in beaver_triplet_shares]
        
        # 计算E和F的share
        E_shares = [A_share - P_share for A_share, P_share in zip(A_shares, P_shares)]
        F_shares = [B_share - Q_share for B_share, Q_share in zip(B_shares, Q_shares)]
        
        # 从share中恢复E和F
        E = np.sum(E_shares, axis=0)
        F = np.sum(F_shares, axis=0)
        
        # 计算G、S和T的share
        G_shares = []
        S_shares = []
        T_shares = []
        
        for i in range(n):
            if i == 0:
                G_shares.append(np.matmul(E, F))
                S_shares.append(np.matmul(E, Q_shares[i]))
                T_shares.append(np.matmul(P_shares[i], F))
            else:
                G_shares.append(np.zeros_like(G_shares[0]))
                S_shares.append(np.matmul(E, Q_shares[i]))
                T_shares.append(np.matmul(P_shares[i], F))
        
        # 计算C的share
        C_shares = []
        for i in range(n):
            C_share = G_shares[i] + S_shares[i] + T_shares[i] + O_shares[i]
            C_shares.append(C_share)
            
        return C_shares
    
    @staticmethod
    def print_matrix(matrix, title=None):
        """
        美观地打印矩阵
        :param matrix: 要打印的矩阵
        :param title: 矩阵的标题
        """
        if title:
            print(title)
        if isinstance(matrix, np.ndarray):
            for row in matrix:
                print(" ".join(f"{val:.2f}" for val in row))
        else:
            for row in matrix:
                print(" ".join(f"{val:.2f}" for val in row))
        print()

if __name__ == "__main__":

    # 生成Beaver三元组
    A, B, C = SecretShare.generate_beaver_triplet(2, 3, 2)
    
    # 打印生成的Beaver三元组
    SecretShare.print_matrix(A, "Matrix A")
    SecretShare.print_matrix(B, "Matrix B")
    SecretShare.print_matrix(C, "Matrix C (A @ B)")
    
    # 将矩阵A和B分成3个部分
    A_shares = SecretShare.secret_share_matrix(A, 3)
    B_shares = SecretShare.secret_share_matrix(B, 3)
    C_shares = SecretShare.secret_share_matrix(C, 3)
    
    # 打印分成的部分
    for i, share in enumerate(A_shares):
        SecretShare.print_matrix(share, f"Matrix A Share {i+1}")
    for i, share in enumerate(B_shares):
        SecretShare.print_matrix(share, f"Matrix B Share {i+1}")
    
    # 生成矩阵X和Y，维度与A和B保持一致
    X = np.random.uniform(0, 10, A.shape)
    Y = np.random.uniform(0, 10, B.shape)
    
    # 打印生成的矩阵X和Y
    SecretShare.print_matrix(X, "Matrix X")
    SecretShare.print_matrix(Y, "Matrix Y")
    
    # 计算矩阵X和Y相乘的结果Z
    Z = np.matmul(X, Y)
    
    # 打印结果矩阵Z
    SecretShare.print_matrix(Z, "Matrix Z (X @ Y)")

    X_shares = SecretShare.secret_share_matrix(X, 3)
    Y_shares = SecretShare.secret_share_matrix(Y, 3)

    # 生成Beaver三元组的共享部分
    beaver_triplet_shares = []
    for i in range(3):
        a_share = A_shares[i]
        b_share = B_shares[i]
        c_share = C_shares[i]
        beaver_triplet_shares.append((a_share, b_share, c_share))
    
    # 使用Beaver三元组进行矩阵乘法
    Z_shares = SecretShare.beaver_multiply(X_shares, Y_shares, beaver_triplet_shares)
    
    # 打印结果共享部分
    for i, share in enumerate(Z_shares):
        SecretShare.print_matrix(share, f"Matrix Z Share {i+1}")
    
    # 还原矩阵Z
    Z_reconstructed = SecretShare.reconstruct_matrix(Z_shares)
    SecretShare.print_matrix(Z_reconstructed, "Reconstructed Matrix Z")