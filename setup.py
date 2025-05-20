from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='DNACipher',
    version='1.0.2',
    author='Brad Balderson',
    author_email='bbalderson@salk.edu',
    packages=find_packages(),
    include_package_data=True,  # <<< this is crucial
    #license=TBD #'GPL-3.0',
    long_description_content_type="text/markdown",
    long_description=long_description,
    scripts=['bin/dnacipher'],
    install_requires = ['pandas', "numpy"],
    entry_points={
        'console_scripts': [
        'dnacipher=dnacipher.main:main',
        ]
    },
    python_requires='>=3',
    description=("DNACipher predicts the effects of genetic variants across observed and unobserved biological contexts, enabling comprehensive deep variant impact mapping of GWAS signals."),
    keywords=['DNACipher', 'dnacipher', 'DeepVariantImpactMapping', 'DVIM'],
    url='https://github.com/BradBalderson/DNACipher',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Bio-Informatics :: DeepLearning",
        # license TBD "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    ],
)