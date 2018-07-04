from setuptools import setup
from os import path

with open(path.join(path.abspath(path.dirname(__file__)), 'README'), encoding='utf-8') as f:
    long_description = f.read()
with open(path.join(path.abspath(path.dirname(__file__)), 'VERSION'), encoding='utf-8') as f:
    version = f.read()
with open(path.join(path.abspath(path.dirname(__file__)), 'requirements.txt'), encoding='utf-8') as f:
    dependencies = f.read()

setup(name='playground',
      packages=['playground'],
      install_requires=dependencies.split('\n'),
      description='A Playground for RL and Robotics Environments',
      long_description=long_description,
      author='Ge Yang',
      url='https://github.com/episodeyang/rl-playground',
      author_email='yangge1987@gmail.com',
      version=version)
