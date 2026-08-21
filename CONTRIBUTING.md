# Contributing to VORaytracer

Thank you for considering a contribution. VORaytracer currently targets Windows, Visual Studio, and NVIDIA RTX hardware.

## Development setup

1. Install the prerequisites documented in `README.md`.
2. Copy `config\Local.props.example` to `config\Local.props` and adjust dependency paths. The local file is intentionally ignored by Git.
3. Build both configurations:

   ```powershell
   .\scripts\Build.ps1 -Configuration Debug
   .\scripts\Build.ps1 -Configuration Release
   ```

4. Run the test executable from each output directory:

   ```powershell
   .\bin\x64\Debug\VORaytracer.Tests.exe
   .\bin\x64\Release\VORaytracer.Tests.exe
   ```

Changes to rendering code should also be checked in both Vulkan and OptiX modes. Include the GPU model, driver version, backend, build configuration, and reproduction steps when reporting rendering defects.

## Pull requests

- Keep each pull request focused on one concern.
- Preserve C++20 compatibility and the existing warning level.
- Update tests and documentation when behavior changes.
- Do not commit SDKs, generated binaries, Visual Studio state, `config\Local.props`, or non-redistributable model/HDR assets.
- Confirm that any new dependency is compatible with the project license and document it in `THIRD_PARTY_NOTICES.md`.

By contributing, you agree that your contribution is licensed under the repository's `LGPL-3.0-or-later` license.
