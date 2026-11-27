from setuptools import setup, find_packages

setup(
    name="hermes-whoosh-memory",
    version="0.3.0",
    description="Local hybrid (BM25 + vector) memory provider for Hermes Agent",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="NousResearch",
    author_email="contact@nousresearch.com",
    url="https://github.com/dorel14/hermes-whoosh-memory",
    project_urls={
        "Bug Tracker": "https://github.com/dorel14/hermes-whoosh-memory/issues",
        "Documentation": "https://github.com/dorel14/hermes-whoosh-memory#readme",
        "Source Code": "https://github.com/dorel14/hermes-whoosh-memory",
    },
    license="MIT",
    packages=find_packages(include=["whoosh_memory*"]),
    py_modules=[],
    package_data={"whoosh_memory": ["plugin.yaml", "README.md"]},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "whoosh-ng>=0.3.0",
    ],
    extras_require={
        "embeddings": ["fastembed>=0.3.0"],
        "sentence-transformers": ["sentence-transformers>=2.5.0"],
        "ollama": ["ollama>=0.1.0"],
        "images": ["Pillow>=10.0.0"],
        "all": [
            "whoosh-ng>=0.3.0",
            "fastembed>=0.3.0",
            "Pillow>=10.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "whoosh-memory=whoosh_memory.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Communications :: Chat",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="hermes memory whoosh search bm25 vector images",
)
