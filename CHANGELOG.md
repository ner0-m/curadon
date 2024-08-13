# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### Added

- ASTRA-like vector geometry description
- 2D geometry can now have changing distances of detector and source

## [0.2.0] - 2024-02-08

### Added

- Use plan to execute projection operations (similar to cuFFT)
- 2D operations support mixed precision data


### Removed

- Texture cache (obsolent with plans)

## [0.1.0] - 2024-02-06

### Added

- 2D and 3D attenuation X-ray CT operations
- Differentiable functions for PyTorch
- Tests based on perceptual hashes
