import pysoem
import time
import struct
import sys


# ============================================================
# USER SETTINGS
# ============================================================

# First run with this set to None.
# The program will print all network adapters.
ADAPTER_NAME = r"\Device\NPF_{6E3791D7-01FB-4DBB-A2A3-528EC9DB113F}"

# Expected StepperOnline A6 identity from your ESI
EXPECTED_VENDOR_ID = 0x00400000
EXPECTED_PRODUCT_CODE = 0x00000715
EXPECTED_REVISION = 0x00002EF8

# VERY SMALL INITIAL TEST MOVE
#
# IMPORTANT:
# This is in DRIVE POSITION UNITS, not degrees yet.
#
# Start with something small like 1000.
# Once we verify the scaling, we can convert degrees -> counts.


# Conservative profile settings.
# These are drive "user units", not necessarily RPM.

PROFILE_ACCELERATION = 50000
PROFILE_DECELERATION = 50000

MOVE_TIMEOUT = 10000.0
COUNTS_PER_OUTPUT_REV = 1310720

def degrees_to_counts(deg):
    return round(deg / 360.0 * COUNTS_PER_OUTPUT_REV)

def rpm_to_velocity(rpm):
    return round(rpm / 60.0 * COUNTS_PER_OUTPUT_REV)

TEST_MOVE = degrees_to_counts(360)
PROFILE_VELOCITY = rpm_to_velocity(50) # RPM

# ============================================================
# CiA402 OBJECTS
# ============================================================

CONTROLWORD = 0x6040
STATUSWORD = 0x6041
MODE_OF_OPERATION = 0x6060
MODE_DISPLAY = 0x6061
ACTUAL_POSITION = 0x6064
TARGET_POSITION = 0x607A

PROFILE_VEL = 0x6081
PROFILE_ACC = 0x6083
PROFILE_DEC = 0x6084

ERROR_CODE = 0x603F


# ============================================================
# SDO HELPERS
# ============================================================

def write_u8(slave, index, value):
    slave.sdo_write(
        index,
        0,
        struct.pack("<B", value)
    )


def write_i8(slave, index, value):
    slave.sdo_write(
        index,
        0,
        struct.pack("<b", value)
    )


def write_u16(slave, index, value):
    slave.sdo_write(
        index,
        0,
        struct.pack("<H", value)
    )


def write_i32(slave, index, value):
    slave.sdo_write(
        index,
        0,
        struct.pack("<i", value)
    )


def write_u32(slave, index, value):
    slave.sdo_write(
        index,
        0,
        struct.pack("<I", value)
    )


def read_u16(slave, index):
    data = slave.sdo_read(index, 0)
    return struct.unpack("<H", data[:2])[0]


def read_i8(slave, index):
    data = slave.sdo_read(index, 0)
    return struct.unpack("<b", data[:1])[0]


def read_i32(slave, index):
    data = slave.sdo_read(index, 0)
    return struct.unpack("<i", data[:4])[0]


# ============================================================
# STATUS HELPERS
# ============================================================

def print_status(slave, label=""):
    status = read_u16(slave, STATUSWORD)
    mode = read_i8(slave, MODE_DISPLAY)
    pos = read_i32(slave, ACTUAL_POSITION)
    error = read_u16(slave, ERROR_CODE)

    print(
        f"{label}"
        f" Statusword=0x{status:04X}"
        f" Mode={mode}"
        f" Position={pos}"
        f" Error=0x{error:04X}"
    )

    return status


def wait_for_status(slave, mask, expected, timeout=3.0):
    start = time.time()

    while time.time() - start < timeout:
        status = read_u16(slave, STATUSWORD)

        if (status & mask) == expected:
            return True

        time.sleep(0.05)

    return False


# ============================================================
# CiA402 DRIVE ENABLE
# ============================================================

def fault_reset(slave):
    status = read_u16(slave, STATUSWORD)

    # CiA402 fault bit = bit 3
    if status & 0x0008:
        print("Drive is faulted. Sending fault reset...")

        write_u16(slave, CONTROLWORD, 0x0080)
        time.sleep(0.2)

        write_u16(slave, CONTROLWORD, 0x0000)
        time.sleep(0.2)

        print_status(slave, "After fault reset:")


def enable_drive(slave):
    print("\nEnabling drive...")

    fault_reset(slave)

    # Shutdown
    print("1. Shutdown")
    write_u16(slave, CONTROLWORD, 0x0006)
    time.sleep(0.2)
    print_status(slave)

    # Status pattern for Ready to Switch On
    if not wait_for_status(slave, 0x006F, 0x0021):
        print("Warning: did not clearly reach Ready to Switch On")

    # Switch On
    print("2. Switch On")
    write_u16(slave, CONTROLWORD, 0x0007)
    time.sleep(0.2)
    print_status(slave)

    # Status pattern for Switched On
    if not wait_for_status(slave, 0x006F, 0x0023):
        print("Warning: did not clearly reach Switched On")

    # Enable Operation
    print("3. Enable Operation")
    write_u16(slave, CONTROLWORD, 0x000F)
    time.sleep(0.3)

    status = print_status(slave)

    # Operation enabled normally appears as x027
    if (status & 0x006F) != 0x0027:
        raise RuntimeError(
            f"Drive did not reach Operation Enabled. "
            f"Statusword = 0x{status:04X}"
        )

    print("Drive is OPERATION ENABLED.")


def disable_drive(slave):
    print("\nDisabling drive...")

    try:
        write_u16(slave, CONTROLWORD, 0x0006)
        time.sleep(0.2)

        write_u16(slave, CONTROLWORD, 0x0000)

    except Exception as e:
        print("Disable warning:", e)


# ============================================================
# PROFILE POSITION MOVE
# ============================================================

def setup_profile_position(slave):
    print("\nSetting Profile Position mode...")

    # 6060 = 1 = Profile Position
    write_i8(slave, MODE_OF_OPERATION, 1)
    time.sleep(0.2)

    mode = read_i8(slave, MODE_DISPLAY)
    print("Mode display:", mode)

    if mode != 1:
        raise RuntimeError(
            f"Drive did not enter Profile Position mode. "
            f"6061 = {mode}"
        )

    print("Setting profile velocity/accel/decel...")

    write_u32(slave, PROFILE_VEL, PROFILE_VELOCITY)
    write_u32(slave, PROFILE_ACC, PROFILE_ACCELERATION)
    write_u32(slave, PROFILE_DEC, PROFILE_DECELERATION)


def move_relative(slave, amount):
    print("\n----------------------------------------")
    print(f"Relative move requested: {amount} units")
    print("----------------------------------------")

    start_position = read_i32(slave, ACTUAL_POSITION)

    print("Starting position:", start_position)

    # Target position is relative displacement because we will use
    # Controlword bit 6 = 1.
    write_i32(slave, TARGET_POSITION, amount)

    # PP mode:
    # 0x000F = operation enabled
    #
    # bit 4 = new set-point
    # bit 5 = change set immediately
    # bit 6 = relative
    #
    # 0x000F
    # +0x0010 new setpoint
    # +0x0020 immediate
    # +0x0040 relative
    # =0x007F

    print("Triggering move...")

    # Make sure new-setpoint bit starts low
    write_u16(slave, CONTROLWORD, 0x006F)
    time.sleep(0.05)

    # Rising edge on bit 4
    write_u16(slave, CONTROLWORD, 0x007F)

    time.sleep(0.1)

    # Return new-setpoint bit low while keeping relative mode
    write_u16(slave, CONTROLWORD, 0x006F)

    start_time = time.time()

    while True:
        status = read_u16(slave, STATUSWORD)
        position = read_i32(slave, ACTUAL_POSITION)
        error = read_u16(slave, ERROR_CODE)

        print(
            f"\rPosition: {position:12d}   "
            f"Status: 0x{status:04X}   "
            f"Error: 0x{error:04X}",
            end="",
            flush=True
        )

        # Fault bit
        if status & 0x0008:
            print()
            raise RuntimeError(
                f"Drive fault during move. Error code 0x{error:04X}"
            )

        # In PP mode Statusword bit 10 indicates target reached
        if status & (1 << 10):
            print()
            print("Target reached.")
            break

        if time.time() - start_time > MOVE_TIMEOUT:
            print()
            raise TimeoutError("Move timed out.")

        time.sleep(0.05)

    final_position = read_i32(slave, ACTUAL_POSITION)

    print("Final position:", final_position)
    print("Moved approximately:", final_position - start_position)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n========================================")
    print("StepperOnline A6 EtherCAT Python Test")
    print("========================================\n")

    print("Available Ethernet adapters:\n")

    adapters = pysoem.find_adapters()

    if not adapters:
        print("No usable Ethernet adapters found.")
        print("Check Npcap installation.")
        sys.exit(1)

    for i, adapter in enumerate(adapters):
        print(f"[{i}] {adapter.name}")
        print(f"    {adapter.desc}")

    # --------------------------------------------------------
    # FIRST-RUN MODE
    # --------------------------------------------------------

    if ADAPTER_NAME is None:
        print("\n----------------------------------------")
        print("ADAPTER_NAME is currently None.")
        print()
        print("Find your Intel I210 in the list above.")
        print("Copy its adapter.name into ADAPTER_NAME")
        print("at the top of this script.")
        print("----------------------------------------")
        return

    master = pysoem.Master()

    try:
        print("\nOpening adapter:")
        print(ADAPTER_NAME)

        master.open(ADAPTER_NAME)

        print("\nScanning EtherCAT network...")

        slave_count = master.config_init()

        if slave_count <= 0:
            raise RuntimeError(
                "No EtherCAT slaves found. "
                "Check CN3 IN, cable, drive power, "
                "and make sure TwinCAT is not using this NIC."
            )

        print(f"Found {slave_count} EtherCAT slave(s).\n")

        for i, s in enumerate(master.slaves):
            print(
                f"Slave {i}: "
                f"name={s.name}, "
                f"vendor=0x{s.man:08X}, "
                f"product=0x{s.id:08X}, "
                f"revision=0x{s.rev:08X}"
            )

        slave = master.slaves[0]

        # ----------------------------------------------------
        # VERIFY A6
        # ----------------------------------------------------

        if slave.man != EXPECTED_VENDOR_ID:
            raise RuntimeError(
                f"Unexpected vendor ID: 0x{slave.man:08X}"
            )

        if slave.id != EXPECTED_PRODUCT_CODE:
            raise RuntimeError(
                f"Unexpected product code: 0x{slave.id:08X}"
            )

        print("\nA6 identity confirmed.")

        if slave.rev != EXPECTED_REVISION:
            print(
                "Warning: revision differs from uploaded ESI:"
                f" expected 0x{EXPECTED_REVISION:08X},"
                f" found 0x{slave.rev:08X}"
            )

        # ----------------------------------------------------
        # INITIAL STATUS
        # ----------------------------------------------------

        print("\nInitial drive state:")
        print_status(slave)

        # ----------------------------------------------------
        # SET PP MODE
        # ----------------------------------------------------

        setup_profile_position(slave)

        # ----------------------------------------------------
        # ENABLE DRIVE
        # ----------------------------------------------------

        enable_drive(slave)

        print("\n*** MOTOR IS NOW ENABLED ***")
        print("*** SHAFT MAY PRODUCE TORQUE ***")

        time.sleep(1)

        # ----------------------------------------------------
        # SMALL TEST MOVE
        # ----------------------------------------------------

        move_relative(slave, TEST_MOVE)

        print("\nTest complete.")

    except KeyboardInterrupt:
        print("\nCTRL+C detected.")

    except Exception as e:
        print("\nERROR:")
        print(e)

    finally:
        if master.slaves:
            try:
                disable_drive(master.slaves[0])
            except Exception:
                pass

        try:
            master.close()
        except Exception:
            pass

        print("\nEtherCAT interface closed.")


if __name__ == "__main__":
    main()