<div align="center">
  <img src="PBSP_logo.png" height=100>
</div>

[![DOI](Pending)](Pending)

# SIMUnet: Leveraging open-source machine learning to explore the interplay between parton distribution functions and potential new physics

[PBSP](https://www.pbsp.org.uk/) (Physics Beyond the Standard Proton) is an ERC funded project, led by Prof. Maria Ubiali and based at the Department of Applied Mathematics and Theoretical Physics at the University of Cambridge. The projects focuses on the global interpretation of the LHC data in terms of indirect searches for new physics, by providing a robust framework to globally interpret all subtle deviations from the SM predictions that might arise at colliders.

The PBSP team has developed the SIMUnet methodology, which uses machine learning techniques to study the interplay between PDFs and potential new physics signals. Drawing upon
the [NNPDF methodology](https://arxiv.org/abs/2109.02653), SIMUnet provides an augmented framework with a suite of tools that allows the user to

- Perform simultaneous fits of PDFs and EFT coefficients
- Perform Fixed-PDF fits of EFT coefficients
- Assess the possible absorption of new physics by the PDFs
- Study the interplay between PDFs and EFT coefficients 
- Analyse the results and produce posterior distributions, correlations, confidence levels, and general quality metrics and plots

## Documentation

The documentation is available at the official [SIMUnet website](https://hep-pbsp.github.io/SIMUnet/sphinx/build/html/index.html).

## Install

See the [SIMUnet installation tutorial](https://hep-pbsp.github.io/SIMUnet/sphinx/build/html/tutorials/Installation.html).

## Cite

The SIMUnet code has been developed in [the original paper](https://inspirehep.net/literature/2013000):

```bibtex
@article{Iranipour:2022iak,
    author = "Iranipour, Shayan and Ubiali, Maria",
    title = "{A new generation of simultaneous fits to LHC data using deep learning}",
    eprint = "2201.07240",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.1007/JHEP05(2022)032",
    journal = "JHEP",
    volume = "05",
    pages = "032",
    year = "2022"
}
```

and made public in the [official release](https://inspirehep.net/literature/2755426):

```bibtex
@article{Costantini:2024xae,
    author = "Costantini, Mark N. and Hammou, Elie and Kassabov, Zahari and Madigan, Maeve and Mantani, 
    Luca and Morales Alvarado, Manuel and Moore, James M. and Ubiali, Maria",
    title = "{SIMUnet: an open-source tool for simultaneous global fits of EFT Wilson coefficients and PDFs}",
    eprint = "2402.03308",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "2",
    year = "2024"
}
```

It is directly based on the [NNPDF open-source code](https://arxiv.org/abs/2109.02671):

```bibtex
@article{NNPDF:2021uiq,
    author = "Ball, Richard D. and others",
    collaboration = "NNPDF",
    title = "{An open-source machine learning framework for global analyses of parton distributions}",
    eprint = "2109.02671",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "Edinburgh 2021/13, Nikhef-2021-020, TIF-UNIMI-2021-12",
    doi = "10.1140/epjc/s10052-021-09747-9",
    journal = "Eur. Phys. J. C",
    volume = "81",
    number = "10",
    pages = "958",
    year = "2021"
}
```

The physics potential of the tool has been explored in:

```bibtex
@article{Kassabov:2023hbm,
    author = "Kassabov, Zahari and Madigan, Maeve and Mantani, Luca and Moore, James and Morales Alvarado, 
    Manuel and Rojo, Juan and Ubiali, Maria",
    title = "{The top quark legacy of the LHC Run II for PDF and SMEFT analyses}",
    eprint = "2303.06159",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    doi = "10.1007/JHEP05(2023)205",
    journal = "JHEP",
    volume = "05",
    pages = "205",
    year = "2023"
}

@article{Hammou:2023heg,
    author = "Hammou, Elie and Kassabov, Zahari and Madigan, Maeve and Mangano, Michelangelo L. and 
    Mantani, Luca and Moore, James and Alvarado, Manuel Morales and Ubiali, Maria",
    title = "{Hide and seek: how PDFs can conceal new physics}",
    eprint = "2307.10370",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "CERN-TH-2023-137",
    doi = "10.1007/JHEP11(2023)090",
    journal = "JHEP",
    volume = "11",
    pages = "090",
    year = "2023"
}
```

Please consider citing these papers if you use the code.

## Bugs and contributions

If you find a bug or have a new feature idea, do not hesitate to drop them in our [issue tracker](https://github.com/HEP-PBSP/SIMUnet/issues).
