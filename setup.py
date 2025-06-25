from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="zenllm",
    version="0.1.6",
    author="Koen van Eijk",
    description="A zen, simple, and unified API to prompt LLMs from Anthropic, Google, OpenAI, and more, using only the requests library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://koenvaneijk.com",
    packages=find_packages(),
    install_requires=[
        "requests",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)