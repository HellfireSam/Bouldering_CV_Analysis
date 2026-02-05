from setuptools import setup, find_packages


setup(
	name="bouldering-cv-analysis",
	version="0.1.0",
	description="Personal project to help analyse performance in bouldering.",
	package_dir={"": "src"},
	packages=find_packages("src"),
	include_package_data=True,
	python_requires=">=3.10",
)
