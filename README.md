# EpiHealthForecast: Deep Learning for COVID-19 Hospitalization Forecasting

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FICMLA58977.2023.00203-blue)](https://doi.org/10.1109/ICMLA58977.2023.00203)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository contains the implementation and experimental code for the research paper:

> **Deep Learning Based Forecasting of COVID-19 Hospitalisation in England: A Comparative Analysis**
>
> Michael Ajao-olarinoye, Vasile Palade, Seyed Mousavi, Fei He, and Petra A. Wark
>
> *2023 International Conference on Machine Learning and Applications (ICMLA)*, pp. 1344-1349
>
> Jacksonville, FL, USA | December 15-17, 2023
>
> DOI: [10.1109/ICMLA58977.2023.00203](https://doi.org/10.1109/ICMLA58977.2023.00203)

### Abstract

In the midst of the COVID-19 pandemic, it was essential to accurately forecast the demand for hospitalisation resources to achieve an effective allocation of healthcare resources. This paper explores the potential of various Deep Learning (DL) models, namely basic Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), Gated Recurrent Units (GRU), Bidirectional RNNs, and Sequence-to-Sequence architectures with the inclusion of attention mechanisms, to forecast the demand for hospitalisation resources (mechanical ventilators) in England during the COVID-19 pandemic. The implementation of simulated annealing (SA) as a hyperparameter tuning method produced certain model structures and good results in terms of prediction accuracy. Our findings show that the LSTM-based models (LSTM_SA), achieved the lowest mean average error (MAE), outperforming other architectures used in this study. The results of this study show the potential of DL models to forecast the demand for resources and could help inform the distribution of hospitalisation resources in England during the COVID-19 pandemic.

- **Recurrent Neural Networks (RNN)**: Vanilla RNN, LSTM, GRU
- **Bidirectional Variants**: BiLSTM, BiGRU  
- **Attention Mechanisms**: Dot Product, General, Additive, and Concat Attention
- **Sequence-to-Sequence Models**: Encoder-Decoder architectures with attention

Our experiments demonstrate that **LSTM models optimized with Simulated Annealing (LSTM_SA)** achieved the lowest Mean Absolute Error (MAE), outperforming other architectures for forecasting ventilator bed occupancy and providing valuable insights for healthcare resource management.

## Key Features

- **Multiple Deep Learning Architectures**: Implementation of RNN, LSTM, GRU, and their bidirectional variants
- **Attention Mechanisms**: Four types of attention (Dot Product, General, Additive, Concat)
- **Multi-horizon Forecasting**: Support for 1-day to 14-day ahead predictions
- **Comprehensive Evaluation**: MAE, RMSE, MAPE metrics across multiple forecast horizons
- **NHS England Data Pipeline**: Complete data preprocessing for UK COVID-19 healthcare data

<!-- ## Repository Structure

```
EpiHealthForecast/
├── 📁 data/
│   ├── raw/                    # Raw COVID-19 data from NHS England
│   ├── interim/                # Intermediate processed data
│   ├── processed/              # Final preprocessed datasets
│   └── region_daily_data/      # Regional breakdown data
├── 📁 notebooks/
│   ├── 00_Exploratory_Analysis.ipynb     # Data exploration and visualization
│   ├── 01_Baseline_Comparison.ipynb      # Model comparison experiments
│   ├── 02_Novel_Model_Implementation.ipynb # Advanced model implementations
│   ├── 03_Hyperparameter_Tuning.ipynb    # Hyperparameter optimization
│   ├── experiment1.ipynb                 # Simulated Annealing experiments (MAIN)
│   ├── Data_preprocess.ipynb             # Data preprocessing pipeline
│   ├── forecasting.ipynb                 # Core forecasting experiments
│   └── maps.ipynb                        # Geographic visualizations
├── 📁 src/
│   ├── dl/                     # Deep learning models
│   │   ├── attention.py        # Attention mechanism implementations
│   │   ├── models.py           # Base model classes
│   │   ├── multivariate_models.py  # RNN/LSTM/GRU implementations
│   │   ├── dataloaders.py      # PyTorch data loaders
│   │   ├── informer.py         # Informer architecture
│   │   └── autoformer.py       # Autoformer architecture
│   ├── transforms/             # Data transformations
│   │   ├── stationary_utils.py # Stationarity transformations
│   │   └── target_transformations.py  # Target variable transforms
│   ├── decomposition/          # Time series decomposition
│   │   └── seasonal.py         # Seasonal decomposition
│   ├── outliers/               # Outlier detection
│   │   └── outlier_detection.py
│   └── utils/                  # Utility functions
│       ├── data_utils.py       # Data processing utilities
│       ├── plotting_utils.py   # Publication-quality visualizations
│       ├── ts_utils.py         # Time series utilities
│       └── general.py          # General helper functions
├── 📁 scripts/
│   └── data.py                 # Data loading from NHS API
├── 📁 figures/                 # Generated figures and visualizations
├── 📁 models/                  # Saved model checkpoints
├── 📁 configs/                 # Configuration files
├── 📁 report/                  # Research reports and documentation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
``` -->
<!-- 
## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/EpiHealthForecast.git
   cd EpiHealthForecast
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch with CUDA support** (optional, for GPU acceleration)
   ```bash
   # Visit https://pytorch.org/ for the appropriate command for your system
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage -->

### Data Preparation

The data preprocessing pipeline can be run from the notebooks:

1. **Data Collection**: Run `Data_preprocess.ipynb` to:
   - Fetch COVID-19 data from NHS England API
   - Merge hospitalization, case, and vaccination data
   - Create the final preprocessed dataset

2. **Exploratory Analysis**: Use `00_Exploratory_Analysis.ipynb` for:
   - Data visualization and statistics
   - Time series analysis (trend, seasonality, stationarity)
   - Feature correlation analysis

### Model Training

The main experiments are organized in numbered notebooks:

```python
# Example: Training an LSTM model for ventilator bed occupancy forecasting
from src.dl.multivariate_models import SingleStepRNNConfig, SingleStepRNNModel
from src.dl.dataloaders import TimeSeriesDataModule

# Configure the model
config = SingleStepRNNConfig(
    rnn_type="LSTM",
    input_size=10,      # Number of input features
    hidden_size=64,     # Hidden layer size
    num_layers=2,       # Number of RNN layers
    bidirectional=True, # Use bidirectional LSTM
    learning_rate=1e-3
)

# Create model and train
model = SingleStepRNNModel(config)
```

### Running Experiments

1. **Baseline Comparison**: `01_Baseline_Comparison.ipynb`
   - Compare RNN, LSTM, GRU models
   - Evaluate bidirectional variants
   - Multi-horizon forecasting evaluation

2. **Novel Models**: `02_Novel_Model_Implementation.ipynb`
   - Attention-enhanced models
   - Seq2Seq architectures

3. **Hyperparameter Tuning**: `notebooks/experiment1.ipynb`
   - **Simulated Annealing (SA)** - Primary hyperparameter optimization method
   - Temperature-based acceptance probability with cooling schedule
   - Hyperparameter search space:
     - `rnn_type`: RNN, GRU, LSTM
     - `hidden_size`: 32-128
     - `num_layers`: 5-30
     - `bidirectional`: True/False
   - Models tuned with SA denoted as "LSTM_SA", "GRU_SA", etc.

## Data Description

### Primary Dataset

The study uses COVID-19 healthcare data from NHS England, including:

| Feature | Description |
|---------|-------------|
| `covidOccupiedMVBeds` | **Target variable**: COVID-19 patients on mechanical ventilators |
| `hospitalCases` | Total COVID-19 hospital admissions |
| `newAdmissions` | Daily new hospital admissions |
| `new_confirmed` | Daily confirmed COVID-19 cases |
| `cumAdmissions` | Cumulative hospital admissions |
| `Vax_index` | Vaccination coverage index |

### Time Period

- **Training**: April 2020 - December 2021
- **Validation**: January 2022 - March 2022  
- **Testing**: April 2022 - July 2022

## Model Architectures

### Implemented Models

1. **Vanilla RNN**: Basic recurrent neural network
2. **LSTM**: Long Short-Term Memory networks
3. **GRU**: Gated Recurrent Units
4. **BiLSTM/BiGRU**: Bidirectional variants
5. **Attention-LSTM**: LSTM with attention mechanisms
6. **Seq2Seq**: Encoder-decoder architectures

### Attention Mechanisms

- **Dot Product Attention**: $\text{score}(q, k) = q \cdot k$
- **Scaled Dot Product**: $\text{score}(q, k) = \frac{q \cdot k}{\sqrt{d_k}}$
- **General Attention**: $\text{score}(q, k) = q^T W k$
- **Additive Attention**: $\text{score}(q, k) = v^T \tanh(W_q q + W_k k)$

### Simulated Annealing Hyperparameter Optimization

As described in the published paper, we use **Simulated Annealing (SA)** for hyperparameter tuning. The SA algorithm is a probabilistic optimization technique inspired by the annealing process in metallurgy.

**Algorithm Overview:**

1. **Initialize**: Start with initial hyperparameters and temperature $T_0$
2. **Generate Neighbor**: Randomly perturb current hyperparameters
3. **Evaluate**: Train model and compute MAE on validation set
4. **Accept/Reject**: Accept new solution with probability:
   
   $$P(\text{accept}) = \begin{cases} 1 & \text{if } \Delta E < 0 \\ e^{-|\Delta E|/T} & \text{otherwise} \end{cases}$$
   
5. **Cool**: Reduce temperature: $T_{n+1} = \alpha \cdot T_n$ (cooling rate $\alpha = 0.95$)
6. **Repeat** until convergence or iteration limit

**Implementation Details:**
- Initial temperature: $T_0 = 10$
- Cooling rate: $\alpha = 0.95$
- Maximum iterations: 100
- Early stopping: 5 iterations without improvement
- Objective function: Mean Absolute Error (MAE)

The SA-optimized models (e.g., **LSTM_SA**) achieved the best forecasting performance, as reported in the paper.

<!-- ## Results

### Multi-Horizon Forecasting Performance (MAE)

| Model | 1-day | 3-day | 7-day | 14-day |
|-------|-------|-------|-------|--------|
| RNN | 45.2 | 67.3 | 98.4 | 142.1 |
| LSTM | 38.5 | 54.2 | 82.1 | 118.3 |
| GRU | 39.1 | 55.8 | 84.6 | 121.7 |
| BiLSTM | 35.2 | 49.8 | 76.3 | 108.9 |
| **Attention-LSTM** | **32.4** | **45.6** | **71.2** | **101.4** |

*Note: Results may vary slightly based on random seed and hardware.* -->

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@INPROCEEDINGS{10459821,
  author={Ajao-olarinoye, Michael and Palade, Vasile and Mousavi, Seyed and He, Fei and Wark, Petra A},
  booktitle={2023 International Conference on Machine Learning and Applications (ICMLA)}, 
  title={Deep Learning Based Forecasting of COVID-19 Hospitalisation in England: A Comparative Analysis}, 
  year={2023},
  pages={1344-1349},
  keywords={COVID-19;Deep learning;Ventilators;Recurrent neural networks;Pandemics;Predictive models;Resource management;Deep learning;COVID-19;Hospitalisation forecasting;RNN;LSTM;GRU;Attention mechanism},
  doi={10.1109/ICMLA58977.2023.00203}
}
```

### Recommended Reading

The time series forecasting techniques used in this project were largely informed by the following textbook:

```bibtex
@book{manu_modern_2022,
  title={Modern Time Series Forecasting with Python: Explore Industry-Ready Time Series Forecasting Using Modern Machine Learning and Deep Learning},
  author={Joseph, Manu},
  year={2022},
  edition={1st},
  publisher={Packt Publishing},
  isbn={978-1-80323-204-1}
}
```

This comprehensive guide covers essential topics including ARIMA baselines, feature engineering for time series, LSTM and transformer models, global forecasting paradigms, and multi-step forecasting strategies that were instrumental in developing this research.

<!-- ## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change. -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<!-- ## Acknowledgments

- NHS England for providing COVID-19 healthcare data
- Coventry University for research support
- The PyTorch and PyTorch Lightning teams

## Contact

- **Michael Ajao-olarinoye** - [GitHub](https://github.com/yourusername) | [Email](mailto:your.email@example.com)

---

*This research was conducted as part of PhD research at Coventry University.* -->
