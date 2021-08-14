from setuptools import setup
from setuptools import find_packages


setup(
    name='sklearn-ts',
    packages=find_packages(),
    version='0.0.6',
    description='Package for time series forecasting',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="time series forecasting deep learning machine learning regression ARIMA ETS",
    url='https://github.com/lady-pandas/sklearn-ts',
    author='Marta Markiewicz',
    author_email='m.markiewicz.pl@gmail.com',
    license='MIT',
    zip_safe=False
)
