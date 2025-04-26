
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
 <img src="https://codecov.io/github/yanisvdc/cramR/graph/badge.svg?token=7MX98QJ7Y0" alt="Coverage"/>
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

<br>

##### Overview

- 🎯 **Train & Evaluate ML Models**
- 🔁  **Cram vs. Sample-Splitting & Cross-Validation**
- 🧠 **Typical use case: Policy Learning**
- 🌍 **Real-World Applications**

<br>

##### 🎯 Train & Evaluate ML models

The **Cram method** is an efficient approach to simultaneous learning and evaluation using a generic machine learning (ML) algorithm. 
In a single pass of batched data, the proposed method repeatedly trains an ML algorithm and tests its empirical performance.

##### 🔁 Cram vs. Sample-Splitting & Cross-Validation

Because it utilizes the entire sample for both learning and evaluation, cramming is significantly more data-efficient than sample-splitting, which reserves a portion of the data purely for evaluation.
Also, a key distinction from **cross-validation** is that Cram evaluates the **final learned model** directly, rather than using as a proxy the average performance of multiple fold-specific models trained on different data subsets—resulting in sharper inference and better alignment with real-world deployment.

##### 🧠 Typical use case: Policy Learning

The Cram method naturally applies to the policy learning setting, which is a popular subfield of ML focused on learning a decision rule (also called treatment rule or policy) that assigns treatments or actions to individuals based on their features, with the goal of maximizing an expected outcome (e.g., health, profit, welfare).
Cramming allows users to both learn an individualized decision rule and estimate the average outcome that would result if the learned decision rule were to be deployed to the entire population beyond the data sample.

##### 🌍 Real-World Applications

It is particularly relevant in high-stakes applications where decisions must be both personalized and statistically reliable.

Common examples include:

- **Healthcare**: determining who should receive treatment based on individual characteristics  
- **Advertising and pricing**: setting optimal prices to maximize revenue  
- **Policy interventions**: deciding which individuals or regions should receive targeted support to improve outcomes

---

## 🎯 Key Features

- 🧠 **Cram Policy (`cram_policy`)**: Learn and evaluate individualized binary treatment rules using Cram. Supports flexible models, including causal forests and custom learners. Common examples include whether to treat a patient, send a discount offer, or provide financial aid based on estimated benefit.

- 📈 **Cram ML (`cram_ml`)**: Learn and evaluate standard machine learning models using Cram. It estimates the expected loss at the population level, giving you a reliable measure of how well the final model is likely to generalize to new data. Supports flexible training via caret or custom learners, and allows evaluation with user-defined loss metrics. Ideal for classification, regression, and other predictive tasks.

- 🎰 **Cram Bandit (`cram_bandit`)**: Perform on-policy evaluation of contextual bandit algorithms using Cram. Supports both real data and simulation environments with built-in policies. 

  For users with an ML background, it may be informative to compare with supervised learning to introduce the contextual bandit setting. In supervised learning, each data point comes with a known label. 
  In contrast, the contextual bandit setting involves a context (feature vector), a choice among multiple actions, and a reward observed for the chosen action. 
  Thus, the label (reward) is at first unknown and is only revealed after an action is chosen - note that the labels (rewards) associated with the non-chosen actions will remain unknown (partial feedback), which makes learning and evaluation more challenging. 
  
  Contextual bandits appear in applications where an online system selects actions based on context to maximize outcomes—like showing ads or recommendations and observing user clicks or purchases. Contextual bandit algorithms aim to learn a policy that chooses the best action for each context to maximize expected reward, such as engagement (clicks) or conversion. 
  Cram Bandit estimates how well the final learned policy would perform if deployed on the entire population, based on data collected by the same policy.

---

## 📚 Documentation
- [Introduction & Cram Policy](articles/cram_policy.html)
- [Function Reference](reference/index.html)
- [What's New](news/index.html)

You can also explore additional tutorials and examples through the "Articles" menu in the top navigation bar of the website.

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
    title = {cramR: The Cram Method for Efficient Simultaneous Learning and
Evaluation},
    author = {Yanis Vandecasteele},
    year = {2025},
    note = {R package version 0.1.0, 
https://yanisvdc.github.io/cramR},
    url = {https://github.com/yanisvdc/cramR},
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
