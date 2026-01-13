#!/usr/bin/env python3
"""
Example TCP Client for Blade Detection Data
Connects to the blade detection server and receives real-time data
"""

import socket
import json
import sys
from datetime import datetime


def print_blade_data(data):
    """Pretty print minimal blade detection data"""
    print("\n" + "=" * 50)


    print(f"Move X:   {data.get('x', 0):+.2f} mm")
    print(f"Move Y:   {data.get('y', 0):+.2f} mm")
    print(f"Distance: {data.get('d', 0):.2f} mm")

    print("=" * 50)


def connect_and_receive(host='172.24.9.15', port=5000):
    """Connect to server and receive data"""

    print(f"Connecting to {host}:{port}...")

    try:
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        print(f"✓ Connected to blade detection server!")
        print("Press Ctrl+C to disconnect\n")

        buffer = ""

        while True:
            # Receive data
            data = sock.recv(4096).decode('utf-8')

            if not data:
                print("✗ Connection closed by server")
                break

            buffer += data

            # Process complete JSON messages (newline-delimited)
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)

                if line:
                    try:
                        json_data = json.loads(line)
                        print_blade_data(json_data)

                    except json.JSONDecodeError as e:
                        print(f"✗ JSON decode error: {e}")
                        print(f"  Raw data: {line[:100]}...")

    except ConnectionRefusedError:
        print(f"✗ Could not connect to {host}:{port}")
        print("  Make sure the blade detection server is running")

    except KeyboardInterrupt:
        print("\n\n✓ Disconnected")

    except Exception as e:
        print(f"✗ Error: {e}")

    finally:
        try:
            sock.close()
        except:
            pass


def main():
    """Main entry point"""

    # Parse command line arguments
    if len(sys.argv) == 3:
        host = sys.argv[1]
        port = int(sys.argv[2])
    elif len(sys.argv) == 2:
        host = sys.argv[1]
        port = 5000
    else:
        host = '172.24.9.15'
        port = 5000

    print("=" * 70)
    print("BLADE DETECTION TCP CLIENT (MINIMAL MODE)")
    print("=" * 70)
    print("Receives: Closest blade tip with Y+ offset only")
    print(f"Usage: {sys.argv[0]} [host] [port]")
    print(f"Example: {sys.argv[0]} 192.168.1.100 5000")
    print("=" * 70 + "\n")

    connect_and_receive(host, port)


if __name__ == "__main__":
    main()