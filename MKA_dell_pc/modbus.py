from pymodbus.client import ModbusTcpClient

ROBOT_IP = "172.24.89.89"

client = ModbusTcpClient(ROBOT_IP, port=502)
client.connect()

client.write_register(address=128, value=0)

result = client.read_holding_registers(address=128, count=1)
print(result.registers)

client.close()
