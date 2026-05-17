from setuptools import setup, find_packages

setup(
    name='qtgn-lib',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
    ],
    description='Quantum-Topological Dynamic Graph Networks Library',
    author='Guilherme Bastos',
    author_email='guilhermemfbastos@users.noreply.github.com',
)
