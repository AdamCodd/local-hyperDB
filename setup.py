from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="local-hyperDB",
    version="0.1.5",
    author="Adam F",
    author_email="49641859+AdamCodd@users.noreply.github.com",
    description="A hyper-fast local vector database for use with LLM Agents.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AdamCodd/local-hyperDB/",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pytest",
        "transformers",
        "fast_sentence_transformers"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires='>=3.8',
)
