1. To generate the C++ program for random walk, run the following command :
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) rwcpp.cpp -o rwcpp$(python3-config --extension-suffix)

2. Install all the Python packages in "requirements.txt"

3. Unzip the datasets in the "data" folder

4. Run the code with following command:
python main.py --dataset DATA