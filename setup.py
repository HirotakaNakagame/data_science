import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="HirotakaNakagame", # Replace with your own username
    version="0.0.1",
    author="Hiro nakagame",
    author_email="hirotaka.nakagame@gmail.com",
    description="Useful Tools for Data Science Porjects",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
