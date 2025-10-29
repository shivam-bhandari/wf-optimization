#!/bin/bash

# Code Review Automation Script
# Runs all automated checks for code review

set -e  # Exit on error

echo "=========================================="
echo "  Code Review Automated Checks"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track failures
FAILURES=0

# Function to check command and count failures
check_command() {
    local name=$1
    local command=$2
    
    echo -n "Running ${name}... "
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILURES=$((FAILURES + 1))
        return 1
    fi
}

# Function to run command and show output
run_command() {
    local name=$1
    local command=$2
    
    echo ""
    echo "----------------------------------------"
    echo "  ${name}"
    echo "----------------------------------------"
    eval "$command" || {
        echo -e "${RED}✗ ${name} failed${NC}"
        FAILURES=$((FAILURES + 1))
    }
}

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}Error: Poetry is not installed${NC}"
    echo "Install with: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Check if we're in the project directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: pyproject.toml not found${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

echo "Project: workflow-optimization-benchmark"
echo "Date: $(date)"
echo ""

# 1. Black Format Check
run_command "1. Black Format Check" "poetry run black --check src/ tests/"

# 2. Flake8 Linting
run_command "2. Flake8 Linting" "poetry run flake8 src/ tests/"

# 3. MyPy Type Checking
run_command "3. MyPy Type Checking" "poetry run mypy src/"

# 4. Pytest - Run Tests
run_command "4. Running Tests" "poetry run pytest -v --tb=short"

# 5. Pytest - Coverage Report
run_command "5. Test Coverage" "poetry run pytest --cov=src --cov-report=term --cov-report=html --cov-report=json"

# 6. Poetry Security Audit
run_command "6. Security Audit" "poetry audit"

# Summary
echo ""
echo "=========================================="
echo "  Summary"
echo "=========================================="
echo ""

if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}All automated checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Review the generated coverage report: htmlcov/index.html"
    echo "2. Proceed with manual code review (see docs/CODE_REVIEW_CHECKLIST.md)"
    echo "3. Review architecture and design patterns"
    exit 0
else
    echo -e "${YELLOW}Some checks failed (${FAILURES} failure(s))${NC}"
    echo ""
    echo "Please review the errors above and fix them before proceeding."
    echo "You can still continue with manual review, but address these issues."
    exit 1
fi

