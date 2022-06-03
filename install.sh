echo "Creating the environment 'reinforcedFL' ..."
conda create -n reinforcedFL python=3.6
conda activate reinforcedFL
echo "Environment activated."
echo "Installation of 'pytorch' ..."
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
echo "Installation of requirements ..."
pip install -r requirements.txt
echo "Finished."
