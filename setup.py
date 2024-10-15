from setuptools import setup, find_packages

setup(
    name="fusion_bench",
    version="0.2.0",
    description="A Comprehensive Benchmark of Deep Model Fusion",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Anke Tang",
    author_email="tang.anke@foxmail.com",
    url="https://github.com/tanganke/fusion_bench",
    project_urls={
        "Repository": "https://github.com/tanganke/fusion_bench",
        "Homepage": "https://github.com/tanganke/fusion_bench",
        "Issues": "https://github.com/tanganke/fusion_bench/issues",
    },
    license="LICENSE",
    python_requires=">=3.10",
    install_requires=[
        "hydra-core",
        "torch>=2.0.0",
        "lightning",
        "transformers",
        "datasets",
        "peft",
        "huggingface_hub",
        "matplotlib",
        "tensorboard",
        "tqdm",
        "rich",
        "scipy",
        "h5py",
        "pytest",
    ],
    keywords=["deep learning", "model fusion", "benchmark"],
    packages=find_packages(where="fusion_bench"),
    include_package_data=True,
    package_data={"fusion_bench": ["../fusion_bench_config/**/*"]},
    entry_points={
        "console_scripts": [
            "fusion_bench=fusion_bench.scripts.cli:main",
            "fusion_bench_webui=fusion_bench.scripts.webui:main",
        ]
    },
)
