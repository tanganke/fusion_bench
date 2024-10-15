from setuptools import setup, find_packages
import toml

# Load the pyproject.toml file
with open("pyproject.toml", "r") as f:
    pyproject = toml.load(f)

# Extract metadata
project = pyproject["project"]
build_system = pyproject["build-system"]
setuptools_config = pyproject["tool"]["setuptools"]

setup(
    name=project["name"],
    version=project["version"],
    description=project["description"],
    long_description=open(project["readme"]).read(),
    long_description_content_type="text/markdown",
    author=project["authors"][0]["name"],
    author_email=project["authors"][0]["email"],
    url=project["urls"]["Homepage"],
    project_urls={
        "Repository": project["urls"]["Repository"],
        "Issues": project["urls"]["Issues"],
    },
    license=project["license"]["file"],
    python_requires=project["requires-python"],
    install_requires=project["dependencies"],
    keywords=project["keywords"],
    packages=find_packages(where=setuptools_config["package-dir"]["fusion_bench"]),
    include_package_data=setuptools_config["include-package-data"],
    package_data={"fusion_bench": setuptools_config["package-data"]["fusion_bench"]},
    entry_points={
        "console_scripts": [
            "fusion_bench=fusion_bench.scripts.cli:main",
            "fusion_bench_webui=fusion_bench.scripts.webui:main",
        ]
    },
)
