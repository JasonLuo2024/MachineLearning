import socket

def start_client(host, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        client_socket.connect((host, port))
        print(f"Connected to {host}:{port}")

        while True:
            message = input("Enter a message (or 'exit' to quit): ")
            if message.lower() == 'exit':
                break
            client_socket.sendall(message.encode())

    except ConnectionRefusedError:
        print("Connection refused. Make sure the server is running.")
    finally:
        client_socket.close()

if __name__ == "__main__":
    host = "127.0.0.1"  # Replace with the IP address of the server
    port = 12345  # Use the same port number you chose for the server
    start_client(host, port)
