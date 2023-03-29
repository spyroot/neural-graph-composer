from setuptools import setup, find_packages

setup_info = dict(
    name='neural_graph_composer',
    version='1.0',
    author='Mustafa Bayramov',
    description='Graph Neural Network Composer',
    license='MIT',
    python_requires='>=3.9',
    author_email='spyroot@gmail.com',
    packages=['neural_graph_composer'] +
             ['neural_graph_composer.' +
              pkg for pkg in find_packages('neural_graph_composer')],
    install_requires=[
        'numpy>=1.23.5',
        'torch>=2.0.0',
        'networkx>=2.9.4',
        'tqdm>=4.62.1',
        'matplotlib>=3.7',
        'scipy~=1.10.1',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    extras_require={
        'dev': [
            'pytest>=3.7',
            'flake8>=3.7',
            'mypy>=0.770',
            'coverage>=5.2',
            'pytest-cov>=2.8',
            'pytest-mock>=3.1',
            'pre-commit>=2.9'
        ]
    },
    keywords=['graph', 'neural network', 'pytorch', 'networkx'],
)

setup(**setup_info)
