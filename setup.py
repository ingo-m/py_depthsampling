"""
py_depthsampling setup.

For development installation:
    pip install -e /path/to/py_depthsampling
"""

from setuptools import setup

setup(name='py_depthsampling',
      version='0.0.1',
      description=(('Analysis and visualisation of cortical depth sampling'
                    + ' results.')),
      url='https://github.com/ingo-m/py_depthsampling',
      # download_url='https://github.com/ingo-m/pyprf/archive/v1.3.11.tar.gz',
      # author='Ingo Marquardt',
      # author_email='ingo.marquardt@gmx.de',
      license='GNU General Public License Version 3',
      install_requires=['numpy', 'scipy', 'nibabel'],
      # setup_requires=['numpy'],
      # keywords=['fMRI'],
      # long_description=long_description,
      packages=['py_depthsampling.boot',
                'py_depthsampling.crf',
                'py_depthsampling.drain_model',
                'py_depthsampling.eccentricity',
                'py_depthsampling.ert',
                'py_depthsampling.get_data',
                'py_depthsampling.main',
                'py_depthsampling.misc',
                'py_depthsampling.permutation',
                'py_depthsampling.plot'],
      )
