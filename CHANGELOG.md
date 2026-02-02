# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Improved Sentinel-1 converter performance
- Acceleration term in `sarkit_convert.create_arp_poly.create_arp_poly`

### Changed
- Dependency versions to more closely match sarkit and spec-0


## [0.2.0] - 2026-01-15

### Added
- Antenna metadata to CSK/CSG and TerraSAR converters
- `sarkit_convert.create_arp_poly` module
- Utility for creating SIDD projection polynomials from GeoTIFF metadata in `sarkit_convert.sidd_metadata`

### Changed
- Renamed packaging extra and module from `tsx` to `terrasar`
- Renamed `sarkit_convert.csk` module to `sarkit_convert.cosmo`
- CLIs tweaked for commonality
- Updated `sarkit` dependency version

### Fixed
- Select metadata handling in Sentinel-1 converter


## [0.1.0] - 2025-07-21

### Added
- Basic converters for CSK/CSG, ICEYE, Sentinel-1, and TerraSAR-X/TanDEM-X

[unreleased]: https://github.com/ValkyrieSystems/sarkit-convert/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ValkyrieSystems/sarkit-convert/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ValkyrieSystems/sarkit-convert/releases/tag/v0.1.0
