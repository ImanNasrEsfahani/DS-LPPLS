# DS-LPPLS Model for Financial Bubble Detection

This repository implements the DS-LPPLS (Diagnostic of Super-exponential Log-Periodic Power Law Singularity) model for detecting financial bubbles and crashes in cryptocurrency and traditional markets in real-time.

## Overview

The DS-LPPLS model is a sophisticated tool for detecting and diagnosing financial bubbles in their development phase. It combines:

- Log-Periodic Power Law (LPPL) pattern recognition 
- Multi-scale analysis
- Quantile regression methods
- Ensemble forecasting techniques

The model generates two key indicators:

- **DS LPPLS Confidence™**: Measures the sensitivity of bubble patterns across different time windows
- **DS LPPLS Trust™**: Quantifies how well the theoretical LPPL model matches empirical price data

## Key Features

- Real-time bubble detection in cryptocurrency and traditional markets
- Robust pattern recognition across multiple time scales
- Ensemble forecasting to improve prediction reliability
- Both positive (price increase) and negative (price decrease) bubble detection
- Detailed visualization tools for analysis

## Installation

1. Clone this repository:
```bash
git clone https://github.com/ImanNasrEsfahani/DS-LPPLS
cd DS-LPPLS
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main program:
```bash
python setup.py
```

Follow the interactive prompts to:
1. Select a currency pair/asset
2. Set date range
3. Choose operations:
   - Download data
   - Run model
   - Apply filters
   - Display charts

Available visualization options:
1. Confidence and Trust indicators
2. Simulated Price trajectories  
3. Critical Time predictions

## System Requirements

- Python 3.10+
- Recommended: 16+ CPU cores for optimal performance
- Sufficient storage space (1+ GB per year of data analyzed)
- Minimum 16GB RAM recommended
- Minimum 20GB HDD free space recommended

## Data Storage

- Model results are saved in chunks of 250 rolling windows
- Each chunk is approximately 500-750MB
- Results stored in both .joblib and .csv formats
- Expect ~1GB storage per year of analyzed data

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

For major changes, please open an issue first to discuss proposed changes.

## License

[MIT License](LICENSE)

## Support and Contact

If you have questions, need help, or want to contribute to this project, feel free to:

- Open an [Issue](https://github.com/ImanNasrEsfahani/DS-LPPLS/issues) for bugs, feature requests, or questions
- Contact me directly at  <a href="mailto:Contact@ImanNasr.com">[📧 Email:Contact]</a>
- Website: <a href="https://www.ImanNasr.com">[🌐 Website:ImanNasr.com]</a>

I welcome feedback and contributions to improve this tool for detecting financial bubbles. Whether you want to report a bug, request a feature, or collaborate on development, I'm happy to help!


Citations:

1. [Early Warning Signals of Financial Crises](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/44120409/30012791-4c1d-4ace-bf88-a3aa2931600b/Early-Warning-Signals-of-Financial-Crises.pdf) [[1]]
2. [Everything You Always Wanted to Know About LPPLS](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/44120409/5edf7b51-7082-46f5-8118-b609568186cc/Everything-You-Always-Wanted-to-Know.pdf) [[2]]
3. [MPRA Paper on Log-Periodic Power Law Models](https://mpra.ub.uni-muenchen.de/47869/1/MPRA_paper_47869.pdf) [[3]]
4. [Log-Periodic Power Law Analysis in Financial Markets](https://abis-files.gazi.edu.tr/avesis/6d6a7a02-42b0-44f6-bb8b-b995a39fc022?AWSAccessKeyId=XSO45GTNG2LKZD8YO90K&Expires=1701176902&Signature=hSqawB5FTHPAc74NQ0OS9TD9V6M%3D) [[4]]
5. [ArXiv Paper on DS-LPPLS Model Applications](https://arxiv.org/html/2405.12803v1) [[5]]
6. [University of Pretoria Working Paper on Financial Bubbles](https://www.up.ac.za/media/shared/61/WP/wp_2016_06.zp78711.pdf) [[6]]
7. [SSRN Paper on Bubble Detection Techniques](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4734944) [[7]]
8. [LPPLS Appliance GitHub Repository](https://github.com/sabato96/LPPLS-APPLIANCE) [[8]]
9. [Imperial College London Thesis on Financial Risk Modeling](https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/marc_jeremy_01795865.pdf) [[9]]
10. [Politecnico di Milano Thesis on LPPLS for Bubble Detection](https://www.politesi.polimi.it/retrieve/c280fa02-ead6-4885-8602-3f6c24dfb6d1/Bonanomi%20-%20Log%20Periodic%20Power%20Law%20model%20for%20the%20detection%20of%20financial%20bubbles.pdf) [[10]]
11. [PyPI Package for LPPLS Implementation](https://pypi.org/project/lppls) [[11]]
12. [ETH Zurich Dissertation on Entrepreneurial Risks](https://ethz.ch/content/dam/ethz/special-interest/mtec/chair-of-entrepreneurial-risks-dam/documents/dissertation/thesis_jgerlach_final_202103.pdf) [[12]]
13. [Wiley Article on LPPLS Methodology](https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/wics.1649) [[13]]
14. [ResearchGate Paper on LPPLS Indicators Over Two Centuries](https://www.researchgate.net/publication/312549574_LPPLS_Bubble_Indicators_over_Two_Centuries_of_the_SP_500_Index) [[14]]
15. [DS-LPPLS Confidence Indicator Visualization](https://www.researchgate.net/figure/DS-LPPLS-End-of-Bubble-signals-and-DS-LPPLS-Confidence-indicator-of-S-P500-monthly-data_fig7_312549574) [[15]]
16. [ETH Zurich FCO Report on Financial Crises](https://ethz.ch/content/dam/ethz/special-interest/mtec/chair-of-entrepreneurial-risks-dam/documents/FCO/FCO_Jan_2022.pdf) [[16]]
17. [Detection of Financial Bubbles Using LPPLS](https://www.researchgate.net/publication/380095450_Detection_of_financial_bubbles_using_a_log-periodic_power_law_singularity_LPPLS_model) [[17]]
18. [Open University Thesis on LPPLS Applications](https://oro.open.ac.uk/73791/1/ChristopherLynchThesisRevised.pdf) [[18]]
    
