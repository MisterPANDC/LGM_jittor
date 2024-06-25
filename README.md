# LGM_jittor
An AIGC Project——3D Gaussian Generation Based on Jittor

# Required Packages
jittor, scipy, kiui, diff-gaussian-rasterization, roma, einops

# How to Run
Train the gaussian model and set the arg `ckpt` to the path of the weight file.
To Inference, follow these steps:

1. Install the required packages: `jittor`, `scipy`, `kiui`, `diff-gaussian-rasterization`, `roma`, `einops`.

2. Inference with the following command:
    ```
    python main.py --model_path=/path/to/model --input_image=/path/to/input_image.jpg --output_image=/path/to/output_image.jpg
    ```
    Replace `/path/to/model` with the path to the trained weight file, `/path/to/input_image.jpg` with the path to the input image, and `/path/to/output_image.jpg` with the desired path for the output image.

    This command will run the inference process using the trained Gaussian model and generate a 3D Gaussian image based on the input image.

3. After running the inference, you can find the generated 3D Gaussian image at the specified output path.

Please make sure to install all the required packages mentioned in the "Required Packages" section before running the inference.
