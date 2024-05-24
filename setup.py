from pathlib import Path

from setuptools import Extension, setup

# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="pymarl_application",
    version="0.0.1",
    author="Dengyu Zhang",
    author_email="zhangdy56@mail2.sysu.edu.cn",
    description="A PyMARL codebase for application.",
    long_description="Aimed at MARL application, based on Extended PyMARL codebase.",
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=['numpy>=1.24',
                      'scipy>=1.10',
                      ],
    packages=['pymarl_application',
              'pymarl_application.utils',
              'pymarl_application.components',
              'pymarl_application.controllers',
              'pymarl_application.envs',
              'pymarl_application.learners',
              'pymarl_application.runners',
              'pymarl_application.pretrained',
              'pymarl_application.modules',
              'pymarl_application.modules.agents',
              'pymarl_application.modules.critics',
              'pymarl_application.modules.mixers',],
    package_dir={'pymarl_application': 'src'},
    package_data={'pymarl_application': ['config/*', 'config/algs/*', 'config/envs/*']}, # ignore pretrained/*.pt
)
