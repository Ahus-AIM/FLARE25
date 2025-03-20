# Some good title
Set up your environment with the following commands:
```
conda create --name medseg python=3.12 
conda activate medseg
pip install light-the-torch && ltt install torch torchvision
pip install torchio matplotlib monai edt medim nibabel connected-components-3d cupy-cuda12x cucim-cu12 pandas
```