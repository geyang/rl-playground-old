from setuptools import setup
from os import path

with open(path.join(path.abspath(path.dirname(__file__)), 'README'), encoding='utf-8') as f:
    long_description = f.read()
with open(path.join(path.abspath(path.dirname(__file__)), 'VERSION'), encoding='utf-8') as f:
    version = f.read()

setup(name='playground',
      packages=['playground'],
      install_requires=[
          # 'gym[mujoco,atari,classic_control,robotics]',
          'mujoco-py',
          'gym',
          'baselines',
          'tqdm',
          'params_proto',
          'ml-logger',
          'moleskin',
          'jayes',
          'waterbear',
          # 'visdom',
          # 'scipy',
          # 'joblib',
          # 'zmq',
          # 'dill',
          # 'progressbar2',
          # 'mpi4py',
          # 'cloudpickle',
          # 'tensorflow>=1.4.0',
          'click',
      ],
      description='A Playground for RL and Robotics Environments',
      long_description=long_description,
      author='Ge Yang',
      url='https://github.com/episodeyang/rl-playground',
      author_email='yangge1987@gmail.com',
      version=version)
