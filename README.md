![python build and test](https://github.com/mspillman/powcodgen/actions/workflows/python-app.yml/badge.svg)
# powcodgen
The code in this repo aims to generate semi-realistic diffraction patterns from the [PowCod database](https://www.ba.ic.cnr.it/softwareic/qualx/powcod-download/) for machine learning applications.

Once you have obtained a copy of the database, you can filter it as desired, using the [notebook](https://nbviewer.org/github/mspillman/powcodgen/blob/main/Filter-PowCod.ipynb) in this repo.

Several method of data augmentation have been implemented:
- variable unit cells to simulate temperature changes (whilst maintaining crystal-system)
- variable peak intensities to simulate preferred orientation (March-Dollase)
- variable Gaussian, Lorentzian and FCJ axial divergence contributions to full Voigt peaks
- variable background profile (Chebyshev polynomials)
- variable background noise
- zero-point errors

[See this post for more information](https://mspillman.github.io/blog/posts/2023-09-08-Generating-synthetic-PXRD-data.html#introduction)
