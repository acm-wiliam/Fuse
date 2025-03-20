import socket
import json
import random


class RSAObliviousTransferBob:
    """
    基于RSA的1-out-of-2选择性传输协议实现 - Bob部分
    Bob想要获取Alice的两个消息中的一个消息mb，而不泄露b的值
    """
    
    def __init__(self):
        """初始化Bob的状态"""
        self.bob_state = None
    
    def choose(self, alice_msg, b):
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
    
    def receive(self, alice_transfer_msg):
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


def connect_to_server(host='localhost', port=12345):
    """连接到Alice服务器并执行选择性传输协议"""
    
    try:
        # 创建套接字
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        print(f"Bob: 已连接到Alice服务器 {host}:{port}")
        
        # 创建OT实例
        bob_ot = RSAObliviousTransferBob()
        
        try:
            # 接收Alice的设置消息
            print("Bob: 等待Alice的设置信息...")
            alice_setup_data = client_socket.recv(4096)
            alice_setup_msg = json.loads(alice_setup_data.decode())
            print("Bob: 收到Alice的设置信息")
            print(f"公钥(e, n): ({alice_setup_msg['e']}, {alice_setup_msg['n']})")
            
            # Bob做出选择
            b = int(input("Bob: 请选择要接收的消息 (0 或 1): "))
            while b not in [0, 1]:
                b = int(input("输入无效，请重新选择 (0 或 1): "))
                
            print(f"Bob: 选择接收消息 m{b}")
            bob_msg = bob_ot.choose(alice_setup_msg, b)
            
            # 发送选择给Alice
            print("Bob: 发送选择信息给Alice...")
            bob_msg_data = json.dumps(bob_msg).encode()
            client_socket.sendall(bob_msg_data)
            
            # 接收Alice的传输消息
            print("Bob: 等待Alice的加密消息...")
            alice_transfer_data = client_socket.recv(4096)
            alice_transfer_msg = json.loads(alice_transfer_data.decode())
            
            # 接收并解密消息
            received_m = bob_ot.receive(alice_transfer_msg)
            
            print("\nBob: 协议完成！")
            print(f"Bob: 接收到的消息m{b} = {received_m}")
            
        except Exception as e:
            print(f"协议执行过程中出错: {e}")
        
    except Exception as e:
        print(f"连接服务器时出错: {e}")
    finally:
        client_socket.close()
        print("连接已关闭")


if __name__ == "__main__":
    connect_to_server()