
<div align="center">
  <img src="man/figures/cram_logo.png" alt="CRAM Logo" width="450" style="margin-bottom: 1.5rem;" />
  <p style="font-size: 1.25rem; max-width: 800px; margin: 0 auto;">
    The Cram Method for Efficient Simultaneous Learning and Evaluation
  </p>
</div>


<p align="center">
  <a href="https://github.com/yanisvdc/cramR">
    <img src="https://img.shields.io/badge/View%20on-GitHub-black?logo=github" alt="View on GitHub">
  </a>
  <a href="https://github.com/yanisvdc/cramR/issues">
    <img src="https://img.shields.io/badge/Report%20a%20Bug-red?logo=bugatti" alt="Report a Bug">
  </a>
  <a href="https://arxiv.org/abs/2403.07031">
    <img src="https://img.shields.io/badge/Read%20Paper-blue?logo=bookstack" alt="Read Paper">
  </a>
</p>

---

## 📚 What is CRAM?

The **CRAM method** is a powerful approach for **simultaneously learning and evaluating decision rules**, such as individualized treatment rules (ITRs), from data.

Unlike traditional approaches like **sample splitting** or **cross-validation**, which waste part of the data on evaluation only, **CRAM reuses all available data** efficiently. 

A key distinction from **cross-validation** is that CRAM evaluates the final learned model, rather than averaging performance across multiple models trained on different data splits.

---

## 🎯 Key Features

- 🧠 **Causal Policy Learning (`cram_policy`)**: Learn and evaluate individualized binary treatment rules using flexible model choices, including causal forests and custom learners — all while efficiently reusing the entire dataset.

- 📈 **Machine Learning Evaluation (`cram_ml`)**: Assess ML models performance using CRAM. Supports flexible model training (via `caret` or user-defined functions) and custom loss functions.

- 🎰 **Bandit Evaluation (`cram_bandit`)**: Perform on-policy evaluation of contextual bandit algorithms using CRAM. Supports both real data and simulation environments with built-in policies.


---

## 📚 Documentation
- [Getting Started](articles/cram_policy.html)
- [Function Reference](reference/index.html)
- [What's New](news/index.html)

---

## 🛠️ Installation

To install the development version of CRAM from GitHub:
```r
# Install devtools if needed
install.packages("devtools")

# Install cramR from GitHub
devtools::install_github("yanisvdc/cramR")
```

---

## 📄 Citation & 🤝 Contributing

### 📚 Citation
If you use CRAM in your research, please cite the following paper:

```bibtex
@article{jia2024cram,
  title={The Cram Method for Efficient Simultaneous Learning and Evaluation},
  author={Jia, Zeyang and Imai, Kosuke and Li, Michael Lingzhi},
  journal={arXiv preprint arXiv:2403.07031},
  year={2024}
}
```

You can also cite the R package:

```bibtex
@Manual{,
  title  = {cramR: The Cram Method for Efficient Simultaneous Learning and Evaluation},
  author = {Yanis Vandecasteele and Michael Lingzhi Li and Kosuke Imai and Zeyang Jia and Longlin Wang},
  year   = {2025},
  note   = {R package version 0.1.0},
  url    = {https://github.com/yanisvdc/cramR}
}
```

### 🤝 How to Contribute
We welcome contributions! To contribute:

```bash
# 1. Fork the repository.

# 2. Create a new branch
git checkout -b feature/your-feature

# 3. Commit your changes
git commit -am 'Add some feature'

# 4. Push to the branch
git push origin feature/your-feature

# 5. Create a pull request.

# 6. Open an Issue or PR at:
# https://github.com/yanisvdc/cramR/issues
```

---

## 📧 Contact
For questions or issues, please [open an issue](https://github.com/yanisvdc/cramR/issues).
