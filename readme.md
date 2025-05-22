# Some good title
Set up your environment with the following commands:
```
conda create --name medseg python=3.12
conda activate medseg
pip install -r requirements.txt
```

Run inference with particular GPU:

```bash
CUDA_VISIBLE_DEVICES=<gpu_number> python -m src.submission.CVPR25_iter_eval_nodocker --test_img_path <path_to_imgs> --validation_gts_path <path_to_gts> --save_path <output_folder> --model_checkpoint <model_weights_path>
```
