from setuptools import setup

setup(
    name="reinforcedFL",
    python_requires="==3.6.*",
    install_requires=[
        "torch==1.10.1",
        "torchvision==0.11.2",
        "pytest==7.0.1",
        "gym==0.21.0",
        "numpy==1.19.5",
        "pygal>=3.0.0",
        "rich-cli>=1.2.2",
    ],
    license="GPL-3.0 license",
    author="Benjamin Bourbon",
    author_email="ben.bourbon06@gmail.com",
    description="Federated Reinforcement Learning project",
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    url="https://github.com/bourbonut/reinforcedFL",
)
