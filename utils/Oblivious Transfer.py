import random
import math
from sympy import mod_inverse

class RSAObliviousTransfer:
    """
    基于RSA的1-out-of-2选择性传输协议实现
    Alice拥有两个消息m0和m1，Bob想要获取其中一个消息mb而不泄露b的值
    Alice也不知道Bob获取了哪一个消息
    """
    
    def __init__(self, key_size=512):
        """初始化参数，设置RSA密钥大小"""
        self.key_size = key_size
    
    def generate_prime(self, bits):
        """生成指定位数的素数"""
        # 简单实现，实际应用中应使用更强的素数生成方法
        while True:
            p = random.getrandbits(bits)
            if p % 2 != 0 and self._is_prime(p):
                return p
    
    def _is_prime(self, n, k=5):
        """Miller-Rabin素性测试"""
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0:
            return False
            
        # 找到 n-1 = 2^r * d 中的 r 和 d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
            
        # 进行k次测试
        for _ in range(k):
            a = random.randint(2, n - 2)
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    def alice_setup(self):
        """
        Alice设置阶段:
        1. 生成RSA密钥对
        2. 生成随机数r0和r1
        """
        # 生成RSA密钥
        p = self.generate_prime(self.key_size // 2)
        q = self.generate_prime(self.key_size // 2)
        n = p * q
        phi = (p - 1) * (q - 1)
        
        # 选择公钥e，一般选择小的素数如65537
        e = 65537
        
        # 计算私钥d
        d = mod_inverse(e, phi)
        
        # 生成随机数r0和r1
        r0 = random.randint(1, n-1)
        r1 = random.randint(1, n-1)
        x0 = r0
        x1 = r1
        
        # 存储Alice的状态
        self.alice_state = {
            'p': p,
            'q': q,
            'n': n,
            'e': e,
            'd': d,
            'r0': r0,
            'r1': r1,
            'x0': x0,
            'x1': x1
        }
        
        # 返回发送给Bob的信息
        return {
            'n': n,
            'e': e,
            'x0': x0,
            'x1': x1
        }
    
    def bob_choose(self, alice_msg, b, m0_size=None, m1_size=None):
        """
        Bob选择阶段:
        1. 选择比特b (0或1)
        2. 生成随机数k
        3. 计算v = (xb + k^e) mod n
        """
        n = alice_msg['n']
        e = alice_msg['e']
        x0 = alice_msg['x0']
        x1 = alice_msg['x1']
        
        # 选择xb
        xb = x0 if b == 0 else x1
        
        # 生成随机数k
        k = random.randint(1, n-1)
        
        # 计算v = (xb + k^e) mod n
        v = (xb + pow(k, e, n)) % n
        
        # 存储Bob的状态
        self.bob_state = {
            'b': b,
            'k': k,
            'n': n
        }
        
        # 返回发送给Alice的信息
        return {
            'v': v
        }
    
    def alice_transfer(self, bob_msg, m0, m1):
        """
        Alice传输阶段:
        1. 计算k0和k1
        2. 计算m'0和m'1
        3. 发送m'0和m'1给Bob
        """
        n = self.alice_state['n']
        d = self.alice_state['d']
        x0 = self.alice_state['x0']
        x1 = self.alice_state['x1']
        v = bob_msg['v']
        
        # 计算k0和k1
        k0 = pow((v - x0) % n, d, n)
        k1 = pow((v - x1) % n, d, n)
        
        # 计算m'0和m'1
        m0_prime = (m0 + k0) % n
        m1_prime = (m1 + k1) % n
        
        # 返回发送给Bob的信息
        return {
            'm0_prime': m0_prime,
            'm1_prime': m1_prime
        }
    
    def bob_receive(self, alice_transfer_msg):
        """
        Bob接收阶段:
        1. 根据b选择m'b
        2. 计算mb = (m'b - k) mod n
        """
        b = self.bob_state['b']
        k = self.bob_state['k']
        n = self.bob_state['n']
        
        # 选择m'b
        mb_prime = alice_transfer_msg['m0_prime'] if b == 0 else alice_transfer_msg['m1_prime']
        
        # 计算mb = (m'b - k) mod n
        mb = (mb_prime - k) % n
        
        return mb


def demonstrate_ot():
    """演示RSA选择性传输协议的使用"""
    # 创建协议实例
    ot = RSAObliviousTransfer(key_size=512)
    
    # Alice设置阶段
    alice_setup_msg = ot.alice_setup()
    print("Alice设置完成，发送给Bob的消息：")
    print(f"公钥(e, n): ({alice_setup_msg['e']}, {alice_setup_msg['n']})")
    print(f"随机数 x0: {alice_setup_msg['x0']}")
    print(f"随机数 x1: {alice_setup_msg['x1']}")
    
    # Bob的选择 (假设Bob想要m1, 即b=1)
    b = 0
    bob_msg = ot.bob_choose(alice_setup_msg, b)
    print("\nBob做出选择 b =", b)
    print(f"Bob发送给Alice的消息 v: {bob_msg['v']}")
    
    # Alice的两个消息
    m0 = 12345  # Alice的第一个消息
    m1 = 67890  # Alice的第二个消息
    print("\nAlice的两个消息:")
    print(f"m0: {m0}")
    print(f"m1: {m1}")
    
    # Alice传输加密后的消息
    alice_transfer_msg = ot.alice_transfer(bob_msg, m0, m1)
    print("\nAlice发送加密后的消息给Bob:")
    print(f"m0_prime: {alice_transfer_msg['m0_prime']}")
    print(f"m1_prime: {alice_transfer_msg['m1_prime']}")
    
    # Bob接收并解密消息
    received_m = ot.bob_receive(alice_transfer_msg)
    print("\nBob解密收到的消息:")
    print(f"接收到的消息: {received_m}")
    print(f"验证结果: Bob{'成功' if received_m == (m0 if b == 0 else m1) else '失败'}接收到了m{b}")


if __name__ == "__main__":
    demonstrate_ot()

