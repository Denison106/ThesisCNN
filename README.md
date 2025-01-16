ThesisCNN is a Python-based project designed to simulate fringe patterns influenced by object properties and train a Convolutional Neural Network (CNN) to predict azimuth angles from these patterns. The project incorporates numerical aperture effects, object phase maps, and fringe simulation to create realistic training data for machine learning models.
This project is ideal for researchers and students working on fringe pattern analysis, phase retrieval, or optical simulations.
Required Python packages:
numpy
matplotlib
tensorflow
scipy

Modules
data_gen.py  Handles fringe pattern simulation and data generation
network.py   Defines the CNN architecture for azimuth angle prediction
utils.py     Provides visualization functions
main.py      Entry point for running the project
    
Place the file containing the phase map into the Data/ directory
Ensure the path to the file in data_gen.py is correct. Update it if necessary
Run the main.py file
