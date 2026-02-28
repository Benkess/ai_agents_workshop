from setuptools import find_packages, setup


setup(
    name="agent_state_obs_api_client",
    version="0.1.0",
    description="Pure Python client for the agent state observation API",
    packages=find_packages(),
    install_requires=["setuptools", "requests"],
    zip_safe=True,
    python_requires=">=3.10",
)
