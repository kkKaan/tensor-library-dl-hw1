
# CerGen Tensor Library

## Introduction

CerGen (short for CENG Gergen – "gergen" being one of the Turkish translations for the term "tensor") is a custom tensor library implemented in Python. The primary goal of this project is to offer a deep understanding and practical experience with tensor operations, foundational to machine learning and deep learning algorithms, without the aid of external libraries such as NumPy or PyTorch.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Examples](#examples)

## Installation

Clone this repository to your local machine using:

```bash
git clone https://github.com/kkKaan/tensor-library-dl-hw1.git
```

Ensure you have Python 3.6 or newer installed on your machine.

## Usage

To use the CerGen library in your project, import the `gergen` class from the `cergen.py` module:

```python
from cergen import gergen
```

You can then instantiate `gergen` objects and utilize the various tensor operations provided by the library.

## Features

- **Tensor Operations**: Includes fundamental tensor operations like addition, subtraction, multiplication, division, and more complex operations such as sine, cosine, and logarithmic functions.
- **Shape Manipulation**: Supports reshaping tensors, calculating transpose, and flattening operations to alter tensor dimensions without changing the underlying data.
- **Norm Calculations**: Offers methods to calculate L1, L2, and Lp norms for tensors.
- **Utility Functions**: Provides utility functions for tensor creation, including random tensor generation with specific shapes and ranges.

## Dependencies

CerGen is designed to be self-contained and does not rely on external libraries for its core functionality. However, for comparison purposes with NumPy in the provided Jupyter Notebook, NumPy must be installed.

## Configuration

No additional configuration is required to use CerGen beyond the initial setup and potential NumPy installation for comparison tests.

## Documentation

For detailed documentation on each class and method within the CerGen library, please refer to the inline comments within the `cergen.py` source code.

## Examples

Examples of how to use the CerGen library, including how to perform various tensor operations and manipulate tensor shapes, can be found in the Jupyter Notebook included with the project. Also, for the further tests of the functionality of the class, you can check test.py, by uncommenting specific parts you may test the methods for edge cases.
