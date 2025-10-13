#!/usr/bin/env python3
"""
Setup script for Vangers Gym Wrapper
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    requirements = [
        'gymnasium>=0.28.0',
        'numpy>=1.20.0',
        'opencv-python>=4.5.0',
        'python-socketio>=5.0.0',
        'websocket-client>=1.0.0',
    ]
    return requirements

setup(
    name='vangers-gym',
    version='0.1.0',
    description='OpenAI Gym environment wrapper for Vangers game engine',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='Vangers Gym Contributors',
    author_email='',
    url='https://github.com/KranX/Vangers',
    license='GPLv3',

    packages=find_packages(),
    py_modules=['vangers_env'],

    python_requires='>=3.7',
    install_requires=read_requirements(),

    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.800',
        ],
        'rl': [
            'stable-baselines3>=1.6.0',
            'torch>=1.10.0',
            'tensorboard>=2.8.0',
        ],
        'ray': [
            'ray[rllib]>=2.0.0',
            'tensorflow>=2.8.0',
        ],
        'viz': [
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'pillow>=8.0.0',
        ]
    },

    entry_points={
        'console_scripts': [
            'vangers-gym-test=examples.python_example:main',
        ],
    },

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    keywords='reinforcement-learning gym environment games ai vangers',

    project_urls={
        'Bug Reports': 'https://github.com/KranX/Vangers/issues',
        'Source': 'https://github.com/KranX/Vangers',
        'Documentation': 'https://github.com/KranX/Vangers/tree/main/gym_wrapper',
    },

    include_package_data=True,
    package_data={
        '': ['*.md', '*.txt', '*.cfg', '*.json', 'libvangers_engine*', 'libvangers_engine.*'],
        'gym_wrapper': ['libvangers_engine*', 'libvangers_engine.*'],
        'examples': ['*.py', '*.cpp'],
    },

    zip_safe=False,
)
