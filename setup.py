from setuptools import setup, find_packages

setup(
    name="zero-cost-tts",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "torchaudio",
        "numpy",
        "scipy",
        "librosa",
        "phonemizer",
        "pytest"
    ],
) 