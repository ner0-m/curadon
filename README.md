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

## Performance

As of now, the 2D forward and backward projection operators perform on a
similar levels as the implementations by TorchRadon, lacking behind a bit
for smaller problems, but outperforming (slightly) for larger problems.

You can see the average operations per second over 50 runs (with 3 warmup runs)
in the following table:

| Benchmark (mean op / s) | curadon      | astra (± vs curadon)  | torch-radon (± vs curadon)    |
|-------------------------|--------------|-----------------------|-------------------------------|
|         forward 2d   64 |     12864.38 |      1162.71 (-91.0%) |             14274.11 (+11.0%) |
|         forward 2d  128 |      7541.54 |      1029.40 (-86.4%) |              8579.41 (+13.8%) |
|         forward 2d  256 |      2922.37 |       722.96 (-75.3%) |              3429.46 (+17.4%) |
|         forward 2d  512 |       869.38 |       363.63 (-58.2%) |              1037.06 (+19.3%) |
|         forward 2d 1024 |       241.41 |       121.98 (-49.5%) |               240.24 ( -0.5%) |
|         forward 2d 2048 |        66.89 |        32.89 (-50.8%) |                47.05 (-29.7%) |
|        backward 2d   64 |     12969.40 |       911.69 (-93.0%) |             15643.38 (+20.6%) |
|        backward 2d  128 |      8666.99 |       682.84 (-92.1%) |              9538.58 (+10.1%) |
|        backward 2d  256 |      3683.74 |       274.31 (-92.6%) |              3640.26 ( -1.2%) |
|        backward 2d  512 |      1083.92 |        81.28 (-92.5%) |              1075.92 ( -0.7%) |
|        backward 2d 1024 |       263.92 |        21.79 (-91.7%) |               266.48 ( +1.0%) |
|        backward 2d 2048 |        66.68 |         5.50 (-91.8%) |                62.46 ( -6.3%) |

The size of the image is given in the description, the detector is of size
$\sqrt{2}n$, where $n$ is the size of the image. You can find the code to run the
benchmarks in the `benchmark` folder.

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
