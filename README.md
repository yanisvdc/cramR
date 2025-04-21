
<div align="center">
  <img src="man/figures/cram_logo.png" alt="CRAM Logo" width="350" style="margin-bottom: 1.5rem;" />
  <p style="font-size: 1.25rem; max-width: 800px; margin: 0 auto;">
    The Cram Method for Efficient Simultaneous Learning and Evaluation
  </p>
</div>


<p align="center">
  <a href="https://github.com/yanisvdc/cramR">
    <img src="https://img.shields.io/badge/View%20on-GitHub-black?logo=github" alt="View on GitHub">
  </a>
  <a href="https://codecov.io/github/yanisvdc/cramR" > 
 <img src="https://codecov.io/github/yanisvdc/cramR/graph/badge.svg?token=7MX98QJ7Y0"/> 
 </a>
  <a href="https://github.com/yanisvdc/cramR/issues">
    <img src="https://img.shields.io/badge/Report%20a%20Bug-red?logo=bugatti" alt="Report a Bug">
  </a>
  <a href="https://www.hbs.edu/ris/Publication%20Files/2403.07031v1_a83462e0-145b-4675-99d5-9754aa65d786.pdf">
    <img src="https://img.shields.io/badge/Read%20Paper-blue?logo=bookstack" alt="Read Paper">
  </a>
</p>

---

## 📚 What is Cram?

The **Cram method** is a powerful approach for **simultaneously learning and evaluating decision rules**—such as individualized treatment rules (ITRs)—from data. It is particularly relevant in high-stakes applications where decisions must be both personalized and statistically reliable.

Common examples include:

- **Healthcare**: determining who should receive treatment based on individual characteristics  
- **Advertising and pricing**: setting optimal prices to maximize revenue  
- **Policy interventions**: deciding which individuals or regions should receive targeted support to improve outcomes

Unlike traditional approaches like **sample splitting** or **cross-validation**, which reserve a portion of the data purely for evaluation, **Cram reuses all available data** efficiently during both training and evaluation.

A key distinction from **cross-validation** is that Cram evaluates the **final learned model** directly, rather than averaging the performance of multiple models trained on different data subsets—resulting in sharper inference and better alignment with real-world deployment.

---

## 🎯 Key Features

- 🧠 **Cram Policy (`cram_policy`)**: Learn and evaluate individualized binary treatment rules using Cram. Offers flexible model choices, including causal forests and custom learners.

- 📈 **Cram ML (`cram_ml`)**: Learn and evaluate ML models using Cram. Supports flexible model training (via `caret` or user-defined functions) and custom loss functions.

- 🎰 **Cram Bandit (`cram_bandit`)**: Learn and perform on-policy evaluation of contextual bandit algorithms using Cram. Supports both real data and simulation environments with built-in policies.


---

## 📚 Documentation
- [Introduction & Cram Policy](articles/cram_policy.html)
- [Function Reference](reference/index.html)
- [What's New](news/index.html)

---

## 🛠️ Installation

To install the development version of Cram from GitHub:
```r
# Install devtools if needed
install.packages("devtools")

# Install cramR from GitHub
devtools::install_github("yanisvdc/cramR")
```

---

## 📄 Citation & 🤝 Contributing

### 📚 Citation
If you use Cram in your research, please cite the following papers:

```bibtex
@techreport{jia2024cram,
  title        = {The Cram Method for Efficient Simultaneous Learning and Evaluation},
  author       = {Jia, Zeyang and Imai, Kosuke and Li, Michael Lingzhi},
  institution  = {Harvard Business School},
  type         = {Working Paper},
  year         = {2024},
  url          = {https://www.hbs.edu/ris/Publication%20Files/2403.07031v1_a83462e0-145b-4675-99d5-9754aa65d786.pdf},
  note         = {Accessed April 2025}
}

```

```bibtex
@misc{jia2025crammingcontextualbanditsonpolicy,
      title={Cramming Contextual Bandits for On-policy Statistical Evaluation}, 
      author={Zeyang Jia and Kosuke Imai and Michael Lingzhi Li},
      year={2025},
      eprint={2403.07031},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2403.07031}, 
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
