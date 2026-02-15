
import sys
import importlib

requirements = [
    "yfinance",
    "pandas",
    "sqlalchemy",
    "psycopg2",
    "dotenv", # python-dotenv is imported as dotenv
    "pytest",
    "pytest_mock",
    "requests",
    "slack_sdk",
    "yaml", # pyyaml is imported as yaml
    "numpy",
    "sec_edgar_downloader",
    "openai",
    "pyrate_limiter",
    "ndg", # ndg-httpsclient
    "OpenSSL", # pyopenssl
    "pyasn1",
    "fredapi",
    "finnhub" # finnhub-python
]

additional_checks = {
    "dotenv": "python-dotenv",
    "yaml": "PyYAML",
    "OpenSSL": "pyOpenSSL",
    "pytest_mock": "pytest-mock",
    "finnhub": "finnhub-python"
}

print(f"Python version: {sys.version}")
print("-" * 30)

failed = []
passed = []

for req in requirements:
    try:
        importlib.import_module(req.replace("-", "_"))
        print(f"✅ {req}: Success")
        passed.append(req)
    except ImportError:
        # Try some common mapping variations if first try fails
        try:
            if req == "dotenv":
                importlib.import_module("dotenv")
            elif req == "pytest_mock":
                importlib.import_module("pytest_mock")
            elif req == "ndg":
                importlib.import_module("ndg.httpsclient")
            else:
                raise ImportError
            print(f"✅ {req}: Success (alternative mapping)")
            passed.append(req)
        except ImportError:
            print(f"❌ {req}: Failed")
            failed.append(req)

print("-" * 30)
print(f"Total: {len(requirements)}, Passed: {len(passed)}, Failed: {len(failed)}")

if failed:
    sys.exit(1)
else:
    sys.exit(0)
