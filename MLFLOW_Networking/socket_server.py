import socket

def start_server(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)  # Listen for one incoming connection

    print(f"Server is listening on {host}:{port}")

    conn, addr = server_socket.accept()
    print(f"Connected to {addr[0]}:{addr[1]}")

    while True:
        data = conn.recv(1024)  # Receive data from the client (max 1024 bytes)
        if not data:
            break
        print(f"Received: {data.decode()}")

    conn.close()

if __name__ == "__main__":
    host = '0.0.0.0'  # Use '0.0.0.0' to listen on all available interfaces
    port = 12345  # Choose a port number (use the same port for the client)
    start_server(host, port)
