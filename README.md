
<div align="center">
  <img src="man/figures/cram_logo.jpg" alt="CRAM Logo" width="300" style="margin-bottom: 1.5rem;" />
  <h1 style="font-size: 2.5rem; margin-top: 0;">CRAM</h1>
  <p style="font-size: 1.25rem; max-width: 800px; margin: 0 auto;">
    The Cram Method for Efficient Simultaneous Learning and Evaluation
  </p>
</div>

<p align="center">
  [![View on GitHub](https://img.shields.io/badge/View%20on-GitHub-black?logo=github)](https://github.com/yanisvdc/cramR)
  [![Report a Bug](https://img.shields.io/badge/Report%20a%20Bug-red?logo=bugatti)](https://github.com/yanisvdc/cramR/issues)
  [![Read Paper](https://img.shields.io/badge/Read%20Paper-blue?logo=bookstack)](https://arxiv.org/abs/2403.07031)
</p>


---

## 📚 What is CRAM?

CRAM provides a statistically rigorous framework that combines **causal inference**, **contextual bandits**, and **machine learning** for optimal policy learning and evaluation.

---

## 🎯 Key Features
- 📊 **Policy Learning:** Develop optimized policies using causal models and contextual bandit approaches.
- ✅ **Robust Evaluation:** Advanced variance estimation using influence function methodology.
- 🎰 **Bandit Framework:** Simulate and optimize policies in dynamic environments.

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

---

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
