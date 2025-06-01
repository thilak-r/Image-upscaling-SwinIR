# 🎨 Unified Artistic Style with Neural Style Transfer & SwinIR Upscaling 🖼️

This project presents a sophisticated pipeline to imbue content images with a consistent artistic aesthetic (specifically an "oil painting" effect) and subsequently upscale them to high resolutions suitable for professional applications using cutting-edge deep learning models.

The core objective is to transform diverse input images (e.g., AI-generated artwork, photographs) by:
1.  Harmonizing their visual style via Neural Style Transfer (NST) 🖌️.
2.  Elevating the resolution of these stylized images for large-format printing or high-detail digital display using SwinIR ✨.

## 🚀 Project Pipeline Stages

1.  **Neural Style Transfer (NST):** This foundational stage artistically merges a content image with a style reference image (e.g., a masterwork by Van Gogh). The output is a novel image that preserves the structural essence of the content while adopting the rich visual characteristics (texture, color palette, brushstrokes) of the style reference.
2.  **Super-Resolution (SwinIR):** The stylized image, typically of moderate resolution post-NST, is then processed by the SwinIR model. This stage intelligently enlarges the image (e.g., 4x magnification), generating a high-resolution rendition ideal for demanding applications like large-scale prints.

## 🌟 Example Results Showcase

The table below offers a comparative view of an original content image and its transformed counterpart after processing through the complete Neural Style Transfer and SwinIR upscaling pipeline.

| Original Content Image (Example Input)                                                     | Stylized & Upscaled Output (NST + SwinIR x4)                                                                   |
| :--------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/thilak-r/Image-upscaling-SwinIR/blob/main/000000293804.jpg?raw=true" alt="Original Content Image" width="400"/> | <img src="https://github.com/thilak-r/Image-upscaling-SwinIR/blob/main/input_img_SwinIR.png?raw=true" alt="Stylized and Upscaled Image" width="400"/> |
| *Input: COCO Dataset Image (`000000293804.jpg`)*                                         | *Output: After applying Van Gogh-inspired "oil painting" style & SwinIR x4 super-resolution*                         |

*(Note: The "Original Content Image" is an example input to the NST process. The "Stylized & Upscaled Output" demonstrates the final result after applying a style from a separate reference image and then upscaling with SwinIR.)*

## 🛠️ Technologies & Libraries

*   Python 3.x
*   PyTorch (Core deep learning framework)
*   TorchVision (For models and image transformations)
*   Pillow (PIL) (Image processing)
*   OpenCV (cv2) (Image processing utilities)
*   NumPy (Numerical operations)
*   Matplotlib (For creating visualizations)
*   `timm` (PyTorch Image Models - a dependency for SwinIR)

## ⚙️ Setup & Operational Workflow

### 1. Neural Style Transfer (NST)

The NST module leverages the VGG19 architecture, pre-trained on ImageNet, for robust content and style feature extraction.

**Primary Script:** `nsp.py` 
**Execution Protocol:**

1.  **Asset Preparation:**
    *   **Content Images:** Curate your source images (photographs, AI art, etc.) into a designated input directory (e.g., `path/to/your/content_images/`).
    *   **Style Image:** Select a high-fidelity style reference image (e.g., a high-resolution scan of an oil painting). Store it in a dedicated directory (e.g., `path/to/your/style_references/`).
2.  **Script Configuration (`nsp.py`):**
    *   Define `style_img_path` and `content_img_path` to point to your chosen style and specific content image, respectively.
    *   Specify `output_dir` for saving the resultant stylized images.
    *   Adjust `imsize` (e.g., 512, 1024, 1280) to set the resolution of the NST output. Higher resolutions provide superior input for the upscaling stage but increase computational demands.
    *   Fine-tune `num_steps`, `style_weight`, and `content_weight` to achieve the desired stylistic balance.
3.  **Initiate NST Process:**
    ```bash
    python nsp.py
    ```
4.  **Output:** Stylized images are archived in the configured output directory.

### 2. High-Resolution Upscaling (SwinIR)

SwinIR is employed to magnify and enhance the stylized images produced by the NST phase.

**Repository Context:** This project may use a local or forked instance of the SwinIR codebase. The canonical SwinIR repository is [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR).

**Primary Script:** `main_test_swinir.py` (located within the SwinIR directory structure)

**Execution Protocol:**

1.  **SwinIR Environment:** Ensure the SwinIR codebase is accessible.
    ```bash
    # If setting up SwinIR for the first time:
    # git clone https://github.com/JingyunLiang/SwinIR.git
    # cd SwinIR
    ```
    *(Note: The `main_test_swinir.py` script used in this project has been adapted to gracefully handle `real_sr` tasks without requiring a ground truth comparison folder.)*
2.  **Input Preparation for SwinIR:**
    *   Establish a dedicated input directory for SwinIR (e.g., `path/to/nst_outputs_for_swinir/`).
    *   Transfer the stylized images from the NST output directory to this new SwinIR input folder.
3.  **Initiate SwinIR Upscaling:**
    *   Navigate to your local SwinIR directory via the terminal.
    *   Execute the command (customize paths and model selection as per your setup):
        ```powershell
        python main_test_swinir.py --task real_sr --scale 4 --model_path model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth --folder_lq "path/to/nst_outputs_for_swinir"
        ```
    *   For memory optimization, particularly with GPUs, append the `--tile <tile_size>` argument (e.g., `--tile 400`).
4.  **Output:** High-resolution upscaled images are stored in a subdirectory within the SwinIR `results` folder (e.g., `SwinIR/results/swinir_real_sr_x4/`).

## 🙏 Acknowledgement

Profound gratitude is extended to **Dr. Victor Sir** for his invaluable guidance, mentorship, and insightful contributions throughout the development of this project.

## 📚 References & Foundational Works

*   **Neural Style Transfer:**
    *   Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image Style Transfer Using Convolutional Neural Networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
*   **SwinIR (Super-Resolution):**
    *   Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., & Timofte, R. (2021). SwinIR: Image Restoration Using Swin Transformer. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops*.
*   **Core Libraries & Datasets:**
    *   PyTorch Official Tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/) (Fundamental for NST implementation).
    *   SwinIR Official GitHub: [https://github.com/JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR).
    *   COCO Dataset: [https://cocodataset.org/](https://cocodataset.org/).
    *   Wikimedia Commons / WikiArt: Resources for sourcing high-quality style reference images.
