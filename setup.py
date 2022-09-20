from setuptools import setup

setup(
    name="easy_transformer_speedy",
    version="0.1.0",
    packages=["easy_transformer_speedy"],
    license="LICENSE",
    description="An implementation of transformers tailored for mechanistic interpretability - modified for faster performance and training.",
    long_description=open("README.md").read(),
    install_requires=[
        "einops",
        "numpy",
        "torch",
        "datasets",
        "transformers",
        "tqdm",
        "pandas",
        "datasets",
        "wandb",
        "triton==2.0.0.dev20220305",  # Triton's latest version on PyPI is not up-to-date
        "plotly",
        "fancy_einsum",
    ],
)
