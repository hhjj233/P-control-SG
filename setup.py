from setuptools import find_packages, setup

setup(
    name='pcontrol',
    version='1.0.0',
    description='Percentile control for highway scenario generation',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Jiaxi Liu, Hang Zhou, Hangyu Li, Yifan Wang, Keke Long, Chengyuan Ma, Bin Ran, Xiaopeng Li',
    url='https://github.com/hhjj233/P-control-SG',
    project_urls={'Project page': 'https://hhjj233.github.io/CornerPercentile/', 'Source': 'https://github.com/hhjj233/P-control-SG'},
    license='MIT',
    packages=find_packages(include=['pcontrol', 'pcontrol.*']),
    python_requires='>=3.10',
    install_requires=['torch>=2.2', 'numpy>=2.0', 'scipy>=1.10', 'pandas>=2.0'],
    extras_require={'classical': ['scikit-learn>=1.3']},
)
