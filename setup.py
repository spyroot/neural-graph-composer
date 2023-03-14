from setuptools import setup, find_packages

setup_info = dict(
    name='neural_graph_composer',
    version='1.0',
    author='Mustafa Bayramov',
    description='Graph Neural Network Composer',
    author_email='spyroot@gmail.com',
    packages=['neural_graph_composer'] +
             ['neural_graph_composer.' +
              pkg for pkg in find_packages('neural_graph_composer')],
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
    }
)

setup(**setup_info)
