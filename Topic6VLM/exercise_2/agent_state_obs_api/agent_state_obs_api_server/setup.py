from setuptools import find_packages, setup


setup(
    name="agent_state_obs_api_server",
    version="0.1.0",
    description="Flask server for the agent state observation API",
    packages=find_packages(
        where="..",
        include=[
            "agent_state_obs_api",
            "agent_state_obs_api.*",
            "agent_state_obs_api_server",
            "agent_state_obs_api_server.*",
            "agent_state_obs_api_agent",
            "agent_state_obs_api_agent.*",
        ],
    ),
    package_dir={"": ".."},
    install_requires=[
        "setuptools",
        "flask",
        "langchain",
        "langchain-core",
        "langchain-openai",
        "langgraph",
        "pydantic",
    ],
    zip_safe=True,
    python_requires=">=3.10",
)
