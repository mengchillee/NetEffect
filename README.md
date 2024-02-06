# NetEffect: Discovery and Exploitation of Generalized Network Effects

------------

Lee, M. C., Shekhar, S., Yoo, J., & Faloutsos, C. (2024, May). NetEffect: Discovery and Exploitation of Generalized Network Effects. *Pacific-Asia Conference on Knowledge Discovery and Data Mining*, 2024.

Extended version with appendix:

https://arxiv.org/abs/2301.00270

Please cite the paper as:

    @inproceedings{lee2021InfoShield,
      title={NetEffect: Discovery and Exploitation of Generalized Network Effects},
      author={Lee, Meng-Chieh and Shekhar, Shubhranshu and Yoo, Jaemin and Faloutsos, Christos},
      booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
      year={2024},
      organization={Springer}
    }

## Data

The edge data for large graphs can be downloaded [here]([https://link-url-here.org](https://drive.google.com/drive/folders/1n9IyxSiyImM4Oa_9WIWL85FHK8PsbDrp?usp=sharing)).

## Usage
1. To generate the C++ program for random walk, run the following command :

`g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) rwcpp.cpp -o rwcpp$(python3-config --extension-suffix)`

2. Install all the Python packages in "requirements.txt":

`pip install -r requirements.txt`

3. Run the code with following command:

`python main.py --dataset DATA`
