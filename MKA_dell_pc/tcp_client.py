#!/usr/bin/env python3
"""
Fixed TCP Client for Blade/Tooth Detection Data
Reads binary protocol: [CMD(1)] + [X(4)] + [Y(4)]
Connects to the detection server and receives real-time data
"""

import socket
import sys
import struct
from datetime import datetime


def print_blade_data(cmd, x_mm=None, y_mm=None):
    """Pretty print valley (middle point) detection data"""
    print("\n" + "=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

    if cmd == 1 and x_mm is not None and y_mm is not None:
        print(f"Status:   VALLEY DETECTED (middle between teeth)")
        print(f"Move X:   {x_mm:+.2f} mm")
        print(f"Move Y:   {y_mm:+.2f} mm")
        distance = (x_mm ** 2 + y_mm ** 2) ** 0.5
        print(f"Distance: {distance:.2f} mm")
    elif cmd == 2:
        print(f"Status:   NO VALLEY DETECTED")
    else:
        print(f"Status:   UNKNOWN COMMAND ({cmd})")

    print("=" * 50)


def connect_and_receive(host='172.24.9.15', port=5000):
    """Connect to server and receive binary data"""

    print(f"Connecting to {host}:{port}...")

    try:
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.connect((host, port))
        print(f"✓ Connected to valley detection server!")
        print("✓ Receiving binary protocol data")
        print("Press Ctrl+C to disconnect\n")

        packet_count = 0

        while True:
            try:
                # Read command byte (1 byte)
                cmd_data = sock.recv(1)

                if not cmd_data:
                    print("✗ Connection closed by server")
                    break

                cmd = int.from_bytes(cmd_data, byteorder='big')

                if cmd == 1:
                    # Valley detected - read 8 more bytes (4 for X, 4 for Y)
                    coord_data = sock.recv(8)

                    if len(coord_data) != 8:
                        print(f"✗ Incomplete data packet (got {len(coord_data)} bytes, expected 8)")
                        continue

                    # Unpack X and Y as signed 32-bit integers (big-endian)
                    x_raw = int.from_bytes(coord_data[0:4], byteorder='big', signed=True)
                    y_raw = int.from_bytes(coord_data[4:8], byteorder='big', signed=True)

                    # Convert from 0.01mm units to mm
                    x_mm = x_raw / 100.0
                    y_mm = y_raw / 100.0

                    packet_count += 1
                    print_blade_data(cmd, x_mm, y_mm)

                elif cmd == 2:
                    # No valley detected
                    packet_count += 1
                    print_blade_data(cmd)

                else:
                    print(f"✗ Unknown command byte: {cmd} (0x{cmd:02X})")

            except struct.error as e:
                print(f"✗ Data unpacking error: {e}")
                continue

    except ConnectionRefusedError:
        print(f"✗ Could not connect to {host}:{port}")
        print("  Make sure the valley detection server is running")

    except KeyboardInterrupt:
        print(f"\n\n✓ Disconnected (received {packet_count} packets)")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

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
    print("VALLEY DETECTION TCP CLIENT (BINARY PROTOCOL)")
    print("=" * 70)
    print("Protocol: [CMD(1 byte)] + [X(4 bytes)] + [Y(4 bytes)]")
    print("  CMD=1: Valley detected (middle point between teeth)")
    print("  CMD=2: No valley detected")
    print("")
    print("Receives: Valley between teeth closest to & above grinder")
    print("")
    print(f"Usage: {sys.argv[0]} [host] [port]")
    print(f"Example: {sys.argv[0]} 192.168.1.100 5000")
    print("=" * 70 + "\n")

    connect_and_receive(host, port)


if __name__ == "__main__":
    main()