from setuptools import setup

setup(
    name="easy_transformer",
    version="0.1.0",
    packages=["easy_transformer"],
    license="LICENSE",
    description="An implementation of transformers tailored for mechanistic interpretability.",
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
        "triton==2.0.0.dev20220305",
        "plotly",
        "fancy_einsum",
    ],
)
