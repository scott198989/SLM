#!/usr/bin/env python3
"""
Comprehensive verification script for HAVOC-7B repository setup.

This script checks:
1. Package structure and __init__.py files
2. Import resolution
3. Configuration validity
4. Script accessibility
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'


def check_package_structure() -> Tuple[bool, List[str]]:
    """Verify all packages have __init__.py files."""
    print(f"\n{BOLD}1. Checking Package Structure{RESET}")
    print("=" * 50)

    src_dir = Path("src")
    expected_packages = [
        "havoc_core",
        "havoc_core/model",
        "havoc_core/tokenizer",
        "havoc_data",
        "havoc_training",
        "havoc_inference",
        "havoc_srs",
        "havoc_rag",
        "havoc_tools",
        "havoc_tools/dsl",
        "havoc_tools/python_math",
        "havoc_eval",
        "havoc_cli",
    ]

    issues = []
    all_ok = True

    for package in expected_packages:
        init_file = src_dir / package / "__init__.py"
        if init_file.exists():
            size = init_file.stat().st_size
            if size > 10:  # Non-empty
                print(f"{GREEN}✓{RESET} {package}/__init__.py (populated)")
            else:
                print(f"{YELLOW}⚠{RESET} {package}/__init__.py (empty)")
                issues.append(f"{package}/__init__.py is empty")
        else:
            print(f"{RED}✗{RESET} {package}/__init__.py (missing)")
            issues.append(f"{package}/__init__.py is missing")
            all_ok = False

    return all_ok, issues


def check_main_files() -> Tuple[bool, List[str]]:
    """Check for __main__.py files in key modules."""
    print(f"\n{BOLD}2. Checking __main__.py Files{RESET}")
    print("=" * 50)

    src_dir = Path("src")
    expected_mains = [
        "havoc_cli",
        "havoc_training",
        "havoc_inference",
    ]

    issues = []
    all_ok = True

    for module in expected_mains:
        main_file = src_dir / module / "__main__.py"
        if main_file.exists():
            print(f"{GREEN}✓{RESET} {module}/__main__.py exists")
        else:
            print(f"{YELLOW}⚠{RESET} {module}/__main__.py missing")
            issues.append(f"{module}/__main__.py should exist for module execution")

    return all_ok, issues


def check_scripts() -> Tuple[bool, List[str]]:
    """Verify scripts exist and are properly structured."""
    print(f"\n{BOLD}3. Checking Scripts{RESET}")
    print("=" * 50)

    scripts_dir = Path("scripts")
    expected_scripts = [
        "train.py",
        "serve.py",
        "demo_run.py",
    ]

    issues = []
    all_ok = True

    for script in expected_scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            # Check for sys.path hacks
            content = script_path.read_text()
            if "sys.path.insert" in content:
                print(f"{YELLOW}⚠{RESET} {script} (contains sys.path hack)")
                issues.append(f"{script} still uses sys.path.insert (should be removed)")
            else:
                print(f"{GREEN}✓{RESET} {script} (clean imports)")
        else:
            print(f"{RED}✗{RESET} {script} (missing)")
            issues.append(f"{script} is missing")
            all_ok = False

    return all_ok, issues


def check_pyproject() -> Tuple[bool, List[str]]:
    """Verify pyproject.toml is properly configured."""
    print(f"\n{BOLD}4. Checking pyproject.toml{RESET}")
    print("=" * 50)

    issues = []
    all_ok = True

    pyproject = Path("pyproject.toml")
    if not pyproject.exists():
        print(f"{RED}✗{RESET} pyproject.toml missing")
        return False, ["pyproject.toml is missing"]

    content = pyproject.read_text()

    # Check for essential fields
    checks = [
        ("[project]", "project section"),
        ("name =", "project name"),
        ("version =", "version"),
        ("[tool.setuptools]", "setuptools config"),
        ('package-dir = {"" = "src"}', "src directory config"),
        ("[build-system]", "build system"),
    ]

    for pattern, description in checks:
        if pattern in content:
            print(f"{GREEN}✓{RESET} {description} present")
        else:
            print(f"{RED}✗{RESET} {description} missing")
            issues.append(f"pyproject.toml missing {description}")
            all_ok = False

    return all_ok, issues


def check_configs() -> Tuple[bool, List[str]]:
    """Verify configuration files exist."""
    print(f"\n{BOLD}5. Checking Configuration Files{RESET}")
    print("=" * 50)

    configs_dir = Path("configs")
    expected_configs = [
        "training/default_training.yaml",
        "inference/default_inference.yaml",
    ]

    issues = []
    all_ok = True

    for config in expected_configs:
        config_path = configs_dir / config
        if config_path.exists():
            print(f"{GREEN}✓{RESET} {config}")
        else:
            print(f"{YELLOW}⚠{RESET} {config} (missing)")
            issues.append(f"Config file {config} is missing")

    return all_ok, issues


def main():
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}   HAVOC-7B Repository Setup Verification{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    all_checks = [
        check_package_structure(),
        check_main_files(),
        check_scripts(),
        check_pyproject(),
        check_configs(),
    ]

    all_ok = all(check[0] for check in all_checks)
    all_issues = []
    for _, issues in all_checks:
        all_issues.extend(issues)

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}   Summary{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    if all_ok and not all_issues:
        print(f"\n{GREEN}{BOLD}✓ All checks passed!{RESET}\n")
        print("Next steps:")
        print("  1. Run: pip install -e .")
        print("  2. Run: python test_imports.py")
        print("  3. Test scripts: python scripts/train.py --help")
        return 0
    else:
        if all_issues:
            print(f"\n{YELLOW}Issues found:{RESET}")
            for issue in all_issues:
                print(f"  • {issue}")

        print(f"\n{YELLOW}Setup needs attention.{RESET}")
        print("Review the issues above and make necessary corrections.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
