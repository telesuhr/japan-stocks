# 古い setuptools (PEP 660 非対応) でも pip install -e を通すためのシム。設定は pyproject.toml 側。
from setuptools import setup

setup(name="jstock", version="0.1.0", packages=["jstock"])
