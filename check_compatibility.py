#!/usr/bin/env python3
"""
CLICK Platform Compatibility Checker
Verifies all dependencies and system requirements for Linux/macOS/Windows.

Usage:
    python check_compatibility.py
"""

import sys
import platform
import subprocess
from importlib import import_module
from typing import Dict, List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check Python version >= 3.8"""
    version = sys.version_info
    if version >= (3, 8):
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (need >= 3.8)"


def check_module(module_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a Python module is installed and get its version."""
    import_name = import_name or module_name

    try:
        mod = import_module(import_name)
        version = getattr(mod, '__version__', 'unknown')

        # Special case for pylsl
        if import_name == 'pylsl':
            try:
                version = mod.version_info()
            except:
                pass

        return True, f"{module_name} {version}"
    except ImportError:
        return False, f"{module_name} NOT INSTALLED"


def check_bluetooth() -> Tuple[bool, str]:
    """Check Bluetooth availability (platform-specific)"""
    system = platform.system()

    if system == 'Linux':
        try:
            result = subprocess.run(['bluetoothctl', '--version'],
                                   capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                return True, "bluetoothctl available"
            else:
                return False, "bluetoothctl not found"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False, "bluetoothctl not found (install bluez-utils)"

    elif system == 'Darwin':  # macOS
        try:
            result = subprocess.run(['system_profiler', 'SPBluetoothDataType'],
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and 'Bluetooth' in result.stdout:
                return True, "macOS Bluetooth available"
            else:
                return False, "Bluetooth not detected"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False, "system_profiler failed"

    elif system == 'Windows':
        # Windows: bleak handles Bluetooth, just check if service exists
        return True, "Assuming Windows Bluetooth (verify in Settings)"

    else:
        return False, f"Unknown platform: {system}"


def check_display() -> Tuple[bool, str]:
    """Check if display is available for matplotlib"""
    system = platform.system()

    if system == 'Linux':
        display = subprocess.os.environ.get('DISPLAY')
        wayland = subprocess.os.environ.get('WAYLAND_DISPLAY')
        if display or wayland:
            return True, f"Display available ({display or wayland})"
        else:
            return False, "No DISPLAY or WAYLAND_DISPLAY (headless?)"

    elif system == 'Darwin':
        # macOS always has display in GUI mode
        return True, "macOS GUI available"

    elif system == 'Windows':
        return True, "Windows GUI available"

    else:
        return True, "Unknown display status"


def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def print_result(check_name: str, passed: bool, message: str):
    """Print formatted check result"""
    status = "✓" if passed else "✗"
    status_text = "PASS" if passed else "FAIL"
    print(f"  [{status}] {check_name:<30} {status_text:<8} {message}")


def main():
    results: Dict[str, List[Tuple[str, bool, str]]] = {
        'critical': [],
        'required': [],
        'optional': []
    }

    # Header
    print("=" * 70)
    print("CLICK PLATFORM COMPATIBILITY CHECK")
    print("=" * 70)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python: {sys.version}")

    # Critical checks
    print_section("CRITICAL REQUIREMENTS")

    check_name, passed, msg = "Python Version", *check_python_version()
    results['critical'].append((check_name, passed, msg))
    print_result(check_name, passed, msg)

    # Required dependencies
    print_section("REQUIRED DEPENDENCIES")

    required_modules = [
        ('bleak', 'bleak'),
        ('pylsl', 'pylsl'),
        ('matplotlib', 'matplotlib'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
    ]

    for display_name, import_name in required_modules:
        passed, msg = check_module(display_name, import_name)
        results['required'].append((display_name, passed, msg))
        print_result(display_name, passed, msg)

    # Optional dependencies
    print_section("OPTIONAL DEPENDENCIES")

    optional_modules = [
        ('neurokit2', 'neurokit2'),
        ('muselsl', 'muselsl'),
    ]

    for display_name, import_name in optional_modules:
        passed, msg = check_module(display_name, import_name)
        results['optional'].append((display_name, passed, msg))
        print_result(display_name, passed, msg)

    # System requirements
    print_section("SYSTEM REQUIREMENTS")

    bluetooth_name = "Bluetooth Support"
    passed, msg = check_bluetooth()
    results['required'].append((bluetooth_name, passed, msg))
    print_result(bluetooth_name, passed, msg)

    display_name = "Display/GUI"
    passed, msg = check_display()
    results['optional'].append((display_name, passed, msg))
    print_result(display_name, passed, msg)

    # Summary
    print_section("SUMMARY")

    critical_pass = sum(1 for _, p, _ in results['critical'] if p)
    critical_total = len(results['critical'])

    required_pass = sum(1 for _, p, _ in results['required'] if p)
    required_total = len(results['required'])

    optional_pass = sum(1 for _, p, _ in results['optional'] if p)
    optional_total = len(results['optional'])

    print(f"  Critical: {critical_pass}/{critical_total} passed")
    print(f"  Required: {required_pass}/{required_total} passed")
    print(f"  Optional: {optional_pass}/{optional_total} passed")

    # Determine overall status
    all_critical_pass = critical_pass == critical_total
    all_required_pass = required_pass == required_total

    if all_critical_pass and all_required_pass:
        print("\n  ✓ SYSTEM READY - All critical and required checks passed!")
        if optional_pass < optional_total:
            print(f"    ({optional_total - optional_pass} optional features unavailable)")
    elif all_critical_pass:
        print("\n  ⚠ PARTIAL - Critical checks passed, but missing required dependencies")
        print("    Install missing packages with: pip install <package_name>")
    else:
        print("\n  ✗ NOT READY - Critical requirements not met")

    # Platform-specific installation instructions
    if not all_required_pass:
        print_section("INSTALLATION INSTRUCTIONS")

        failed_modules = [name for name, passed, _ in results['required']
                         if not passed and name not in ['Bluetooth Support', 'Display/GUI']]

        if failed_modules:
            print("\n  Install missing Python packages:")
            print(f"    pip install {' '.join(failed_modules)}")

        if platform.system() == 'Darwin':
            print("\n  macOS-specific notes:")
            print("    - Ensure Homebrew installed: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            print("    - Grant Bluetooth permissions: System Settings → Privacy & Security → Bluetooth")
            print("    - Install muselsl: pip install muselsl")

        elif platform.system() == 'Linux':
            print("\n  Linux-specific notes:")
            print("    - Install bluez: sudo apt install bluez bluez-tools")
            print("    - Add user to bluetooth group: sudo usermod -a -G bluetooth $USER")
            print("    - Restart required after group change")

        elif platform.system() == 'Windows':
            print("\n  Windows-specific notes:")
            print("    - Ensure Python from python.org (not Microsoft Store)")
            print("    - Enable Bluetooth in Settings")
            print("    - Install Visual C++ Build Tools if compilation needed")

    print("\n" + "=" * 70)

    # Exit code
    sys.exit(0 if (all_critical_pass and all_required_pass) else 1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCheck interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
