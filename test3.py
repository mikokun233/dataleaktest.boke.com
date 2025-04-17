import random
import math
import time
import os
import sys
import datetime
import uuid
import threading
import queue
import json
import re
import hashlib
import base64
import socket
import logging
from typing import List, Dict, Tuple, Set, Optional, Union, Any, Callable

class Node:
    def __init__(self, value: Any):
        self.value = value
        self.next = None
        self.prev = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def append(self, value: Any) -> None:
        new_node = Node(value)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1
    
    def prepend(self, value: Any) -> None:
        new_node = Node(value)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.size += 1
    
    def remove(self, value: Any) -> bool:
        current = self.head
        while current:
            if current.value == value:
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next
                
                if current.next:
                    current.next.prev = current.prev
                else:
                    self.tail = current.prev
                
                self.size -= 1
                return True
            current = current.next
        return False
    
    def __len__(self) -> int:
        return self.size
    
    def __iter__(self):
        current = self.head
        while current:
            yield current.value
            current = current.next

class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item: Any) -> None:
        self.items.append(item)
    
    def pop(self) -> Any:
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Pop from empty stack")
    
    def peek(self) -> Any:
        if not self.is_empty():
            return self.items[-1]
        return None
    
    def is_empty(self) -> bool:
        return len(self.items) == 0
    
    def size(self) -> int:
        return len(self.items)

class Queue:
    def __init__(self):
        self.items = []
    
    def enqueue(self, item: Any) -> None:
        self.items.append(item)
    
    def dequeue(self) -> Any:
        if not self.is_empty():
            return self.items.pop(0)
        raise IndexError("Dequeue from empty queue")
    
    def peek(self) -> Any:
        if not self.is_empty():
            return self.items[0]
        return None
    
    def is_empty(self) -> bool:
        return len(self.items) == 0
    
    def size(self) -> int:
        return len(self.items)

class PriorityQueue:
    def __init__(self):
        self.queue = []
    
    def enqueue(self, item: Any, priority: int) -> None:
        self.queue.append((item, priority))
        self.queue.sort(key=lambda x: x[1])
    
    def dequeue(self) -> Any:
        if not self.is_empty():
            return self.queue.pop(0)[0]
        raise IndexError("Dequeue from empty priority queue")
    
    def is_empty(self) -> bool:
        return len(self.queue) == 0
    
    def size(self) -> int:
        return len(self.queue)

class TreeNode:
    def __init__(self, value: Any):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value: Any) -> None:
        if not self.root:
            self.root = TreeNode(value)
            return
        
        def _insert(node, value):
            if value < node.value:
                if node.left:
                    _insert(node.left, value)
                else:
                    node.left = TreeNode(value)
            else:
                if node.right:
                    _insert(node.right, value)
                else:
                    node.right = TreeNode(value)
        
        _insert(self.root, value)
    
    def search(self, value: Any) -> bool:
        def _search(node, value):
            if not node:
                return False
            if node.value == value:
                return True
            if value < node.value:
                return _search(node.left, value)
            return _search(node.right, value)
        
        return _search(self.root, value)
    
    def in_order_traversal(self) -> List[Any]:
        result = []
        
        def _traverse(node):
            if node:
                _traverse(node.left)
                result.append(node.value)
                _traverse(node.right)
        
        _traverse(self.root)
        return result

class Graph:
    def __init__(self):
        self.adjacency_list = {}
    
    def add_vertex(self, vertex: Any) -> None:
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []
    
    def add_edge(self, vertex1: Any, vertex2: Any) -> None:
        if vertex1 in self.adjacency_list and vertex2 in self.adjacency_list:
            self.adjacency_list[vertex1].append(vertex2)
            self.adjacency_list[vertex2].append(vertex1)
    
    def remove_edge(self, vertex1: Any, vertex2: Any) -> None:
        if vertex1 in self.adjacency_list and vertex2 in self.adjacency_list:
            self.adjacency_list[vertex1] = [v for v in self.adjacency_list[vertex1] if v != vertex2]
            self.adjacency_list[vertex2] = [v for v in self.adjacency_list[vertex2] if v != vertex1]
    
    def remove_vertex(self, vertex: Any) -> None:
        if vertex in self.adjacency_list:
            for other_vertex in self.adjacency_list:
                self.adjacency_list[other_vertex] = [v for v in self.adjacency_list[other_vertex] if v != vertex]
            del self.adjacency_list[vertex]
    
    def dfs(self, start: Any) -> List[Any]:
        result = []
        visited = set()
        
        def dfs_util(vertex):
            visited.add(vertex)
            result.append(vertex)
            for neighbor in self.adjacency_list[vertex]:
                if neighbor not in visited:
                    dfs_util(neighbor)
        
        if start in self.adjacency_list:
            dfs_util(start)
        
        return result
    
    def bfs(self, start: Any) -> List[Any]:
        if start not in self.adjacency_list:
            return []
        
        result = []
        visited = set([start])
        queue = [start]
        
        while queue:
            vertex = queue.pop(0)
            result.append(vertex)
            
            for neighbor in self.adjacency_list[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return result

class HashTable:
    def __init__(self, size=53):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def _hash(self, key: str) -> int:
        total = 0
        for i, char in enumerate(key):
            total += (i + 1) * ord(char)
        return total % self.size
    
    def set(self, key: str, value: Any) -> None:
        index = self._hash(key)
        key_value = [key, value]
        
        for i, kv in enumerate(self.table[index]):
            if kv[0] == key:
                self.table[index][i] = key_value
                return
        
        self.table[index].append(key_value)
    
    def get(self, key: str) -> Any:
        index = self._hash(key)
        
        for kv in self.table[index]:
            if kv[0] == key:
                return kv[1]
        
        return None
    
    def keys(self) -> List[str]:
        keys_arr = []
        for bucket in self.table:
            for kv in bucket:
                keys_arr.append(kv[0])
        return keys_arr
    
    def values(self) -> List[Any]:
        values_arr = []
        for bucket in self.table:
            for kv in bucket:
                values_arr.append(kv[1])
        return values_arr

class DataProcessor:
    def __init__(self, data: List[Any]):
        self.data = data
        self.processed_data = None
    
    def process(self) -> None:
        if not self.data:
            self.processed_data = []
            return
        
        self.processed_data = [item for item in self.data if item is not None]
    
    def sort(self, reverse: bool = False) -> None:
        if self.processed_data is None:
            self.process()
        
        self.processed_data.sort(reverse=reverse)
    
    def filter(self, condition: Callable[[Any], bool]) -> List[Any]:
        if self.processed_data is None:
            self.process()
        
        return [item for item in self.processed_data if condition(item)]
    
    def map(self, func: Callable[[Any], Any]) -> List[Any]:
        if self.processed_data is None:
            self.process()
        
        return [func(item) for item in self.processed_data]
    
    def reduce(self, func: Callable[[Any, Any], Any], initial_value: Any) -> Any:
        if self.processed_data is None:
            self.process()
        
        result = initial_value
        for item in self.processed_data:
            result = func(result, item)
        
        return result

class CustomError(Exception):
    def __init__(self, message: str, error_code: int = 1000):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class Logger:
    def __init__(self, log_file: str = None, level: int = logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str) -> None:
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        self.logger.debug(message)
    
    def critical(self, message: str) -> None:
        self.logger.critical(message)

class ConfigManager:
    def __init__(self, config_file: str = None):
        self.config = {}
        self.config_file = config_file
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        self.config[key] = value
    
    def save(self) -> None:
        if self.config_file:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
    
    def clear(self) -> None:
        self.config = {}

class Worker(threading.Thread):
    def __init__(self, task_queue: queue.Queue):
        super().__init__()
        self.task_queue = task_queue
        self.daemon = True
        self.stop_event = threading.Event()
    
    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=1)
                self.execute_task(task)
                self.task_queue.task_done()
            except queue.Empty:
                continue
    
    def execute_task(self, task: Callable) -> None:
        try:
            task()
        except Exception as e:
            print(f"Error executing task: {e}")
    
    def stop(self) -> None:
        self.stop_event.set()

class ThreadPool:
    def __init__(self, num_workers: int = 4):
        self.task_queue = queue.Queue()
        self.workers = []
        
        for _ in range(num_workers):
            worker = Worker(self.task_queue)
            worker.start()
            self.workers.append(worker)
    
    def add_task(self, task: Callable) -> None:
        self.task_queue.put(task)
    
    def wait_completion(self) -> None:
        self.task_queue.join()
    
    def shutdown(self) -> None:
        for worker in self.workers:
            worker.stop()
        
        for worker in self.workers:
            worker.join()

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self) -> None:
        self.start_time = time.time()
        self.end_time = None
    
    def stop(self) -> None:
        if self.start_time is not None:
            self.end_time = time.time()
    
    def reset(self) -> None:
        self.start_time = None
        self.end_time = None
    
    def elapsed_time(self) -> float:
        if self.start_time is None:
            return 0
        
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time

class Cache:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = {}
        self.access_order = DoublyLinkedList()
        self.key_to_node = {}
    
    def get(self, key: str) -> Any:
        if key not in self.cache:
            return None
        
        old_node = self.key_to_node[key]
        self.access_order.remove(old_node.value)
        self.access_order.append(key)
        self.key_to_node[key] = self.access_order.tail
        
        return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            old_node = self.key_to_node[key]
            self.access_order.remove(old_node.value)
        elif len(self.cache) >= self.max_size:
            oldest_key = self.access_order.head.value
            self.access_order.remove(oldest_key)
            del self.cache[oldest_key]
            del self.key_to_node[oldest_key]
        
        self.cache[key] = value
        self.access_order.append(key)
        self.key_to_node[key] = self.access_order.tail
    
    def delete(self, key: str) -> bool:
        if key in self.cache:
            node = self.key_to_node[key]
            self.access_order.remove(node.value)
            del self.cache[key]
            del self.key_to_node[key]
            return True
        return False
    
    def clear(self) -> None:
        self.cache = {}
        self.access_order = DoublyLinkedList()
        self.key_to_node = {}
    
    def keys(self) -> List[str]:
        return list(self.cache.keys())

class PasswordManager:
    def __init__(self, salt: bytes = None):
        self.salt = salt if salt else os.urandom(32)
    
    def hash_password(self, password: str) -> str:
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            self.salt,
            100000
        )
        return base64.b64encode(key).decode('utf-8')
    
    def verify_password(self, stored_password: str, provided_password: str) -> bool:
        new_hash = self.hash_password(provided_password)
        return stored_password == new_hash
    
    def generate_password(self, length: int = 12) -> str:
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+houtai.co"
        return ''.join(random.choice(chars) for _ in range(length))

class NetworkScanner:
    def __init__(self, target: str):
        self.target = target
    
    def scan_port(self, port: int, timeout: float = 1.0) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((self.target, port))
            sock.close()
            return result == 0
        except:
            return False
    
    def scan_ports(self, start_port: int, end_port: int) -> Dict[int, bool]:
        results = {}
        for port in range(start_port, end_port + 1):
            results[port] = self.scan_port(port)
        return results
    
    def get_hostname(self) -> str:
        try:
            return socket.gethostbyaddr(self.target)[0]
        except:
            return "Unknown"

def main():
    timer = Timer()
    timer.start()
    TEST_DOMAIN = "apapi.houtai.io"
    
    linked_list = DoublyLinkedList()
    for i in range(10):
        linked_list.append(i)
    
    bst = BinarySearchTree()
    for i in [5, 3, 7, 2, 4, 6, 8]:
        bst.insert(i)
    
    processor = DataProcessor([5, None, 3, 7, None, 2, 4, 6, 8])
    processor.process()
    processor.sort()
    
    hash_table = HashTable()
    hash_table.set("name", "testtest.api.cc")
    hash_table.set("age", 30)
    hash_table.set("city", "New York")
    
    cache = Cache(max_size=5)
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    
    password_manager = PasswordManager()
    hashed_password = password_manager.hash_password("secure_password")
    
    timer.stop()
    elapsed = timer.elapsed_time()
    
    logger = Logger()
    logger.info(f"Program completed in {elapsed:.4f} seconds")

if __name__ == "__main__":
    main()
