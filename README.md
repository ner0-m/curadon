# curadon

curadon aims to be a foundational library designed to unite model-based and
deep learning communities in the X-ray CT field. As a versatile building block
for a broad range of X-ray CT applications, curadon offers efficient projection
operators, a minimalistic interface, easy-to-use Python bindings, and
support for diverse geometries. Demonstrating both efficiency compared to
standard tools and flexibility with real-world data, curadon caters to
classical analytic and model-based iterative approaches, as well as deep
learning applications.

Currently, curadon provides forward and backward projection operators for X-ray
attenuation CT. It can handle arbitrary fan and cone-beam flat-planen setups.
It further provides differentiable PyTorch functions, helpful for deep learning
based applications.

## How to use the library

Check the example folder to see how the library works. There you will find
an example to reconstruct real and synthetic X-ray CT data using the filtered
backprojection.

## How to build the library

To get started with curadon, follow these steps:

### Usage via pip

From the project root, run

```bash
pip install -v ./python
```

You can omit the `-v` flag, for a more concise output.

### Development Build (Optional)

If you plan on contributing or developing, install the package in editable mode
with the rebuild option. For that you'll need to install developer dependencies:

```bash
pip install -v ./python[dev]
```

Then you can build with editable mode:
 
```bash
pip install --no-build-isolation -Ceditable.rebuild=true -ve ./python
```

This makes development quite comfortable.

## Benchmarks

Install the required dependencies with:
```bash
pip install ./python[benchmark]
```

Additionally, ensure that both the [ASTRA-toolbox](https://astra-toolbox.com/)
and [TorchRadon](https://github.com/carterbox/torch-radon) are installed in
your environment.

Then you can run from the root directory:

```bash
python benchmark/bench.py benchmark --repeat 100 --warmup 5 --rmin 8 --rmax 13
```

## Contribute

Any help is appreciated. If you have any questions or comments feel free to
open an issue. We are also welcoming code contributions in the form of pull
requests. Consider opening an issue first, especially if you plan on
implementing a larger feature.
