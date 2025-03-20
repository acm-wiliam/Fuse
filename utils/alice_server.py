import socket
import json
import random
import math
from sympy import mod_inverse

class RSAObliviousTransferAlice:
    """
    基于RSA的1-out-of-2选择性传输协议实现 - Alice部分
    Alice拥有两个消息m0和m1，Bob想要获取其中一个消息mb而不泄露b的值
    """
    
    def __init__(self, key_size=512):
        """初始化参数，设置RSA密钥大小"""
        self.key_size = key_size
        self.alice_state = None
    
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
    
    def setup(self):
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
    
    def transfer(self, bob_msg, m0, m1):
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


def start_server(host='localhost', port=12345):
    """启动Alice的服务器"""
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    
    print(f"Alice服务器启动，监听 {host}:{port}")
    
    try:
        while True:
            client_socket, addr = server_socket.accept()
            print(f"连接建立：{addr}")
            
            # 创建OT实例
            alice_ot = RSAObliviousTransferAlice(key_size=512)
            
            try:
                # 设置阶段
                print("Alice: 进行设置...")
                alice_setup_msg = alice_ot.setup()
                setup_data = json.dumps(alice_setup_msg).encode()
                client_socket.sendall(setup_data)
                
                # 接收Bob的选择消息
                print("Alice: 等待Bob的选择...")
                bob_msg_data = client_socket.recv(4096)
                bob_msg = json.loads(bob_msg_data.decode())
                
                # 获取Alice的消息
                m0 = int(input("Alice: 请输入第一个消息 (m0): "))
                m1 = int(input("Alice: 请输入第二个消息 (m1): "))
                print(f"Alice: 设置消息 m0={m0}, m1={m1}")
                
                # 传输阶段
                print("Alice: 传输加密后的消息...")
                alice_transfer_msg = alice_ot.transfer(bob_msg, m0, m1)
                transfer_data = json.dumps(alice_transfer_msg).encode()
                client_socket.sendall(transfer_data)
                
                print("Alice: 传输完成！")
                
            except Exception as e:
                print(f"处理连接时出错: {e}")
            finally:
                client_socket.close()
                print("连接已关闭")
                
    except KeyboardInterrupt:
        print("服务器关闭")
    finally:
        server_socket.close()


if __name__ == "__main__":
    start_server()