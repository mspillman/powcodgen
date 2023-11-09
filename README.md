# powcodgen
The code in this repo aims to generate semi-realistic diffraction patterns from the PowCod database for machine learning applications.

Several method of data augmentation have been implemented:
- variable unit cells to simulate temperature changes (whilst maintaining crystal-system)
- variable peak intensities to simulate preferred orientation (March-Dollase)
- variable Gaussian, Lorentzian and FCJ axial divergence contributions to full Voigt peaks
- variable background profile (Chebyshev polynomials)
- variable background noise
- zero-point errors

[See this post for more information](https://mspillman.github.io/blog/posts/2023-09-08-Generating-synthetic-PXRD-data.html#introduction)

Documentation etc to come in the future
