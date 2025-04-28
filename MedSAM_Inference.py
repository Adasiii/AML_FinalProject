# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F

# visualization functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

def process_folder(image_folder, bbox_folder, mask_output_folder, vis_output_folder):
    # Load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint = "work_dir/MedSAM/medsam_vit_b.pth"
    medsam_model = sam_model_registry["vit_b"](checkpoint=checkpoint)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()

    # Create output folders
    os.makedirs(mask_output_folder, exist_ok=True)
    os.makedirs(vis_output_folder, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.png')]
    
    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            # Read image
            img_path = join(image_folder, img_file)
            img_np = io.imread(img_path)
            
            if len(img_np.shape) == 2:
                img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
            else:
                img_3c = img_np
            
            H, W, _ = img_3c.shape

            # Read bbox from txt file (use first line if multiple bboxes exist)
            bbox_file = join(bbox_folder, os.path.splitext(img_file)[0] + '.txt')
            with open(bbox_file, 'r') as f:
                first_line = f.readline().strip()
                box_np = np.array([int(x) for x in first_line.split()]).reshape(1, 4)

            # Preprocess image
            img_1024 = transform.resize(
                img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )
            img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)

            # Scale bbox to 1024x1024
            box_1024 = box_np / np.array([W, H, W, H]) * 1024

            # Perform segmentation
            with torch.no_grad():
                image_embedding = medsam_model.image_encoder(img_1024_tensor)
                medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)

            # Save mask
            mask_output_path = join(mask_output_folder, f"mask_{img_file}")
            io.imsave(mask_output_path, medsam_seg, check_contrast=False)

            # Save visualization
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(img_3c)
            show_box(box_np[0], ax[0])
            ax[0].set_title("Input Image and Bounding Box")
            ax[1].imshow(img_3c)
            show_mask(medsam_seg, ax[1])
            show_box(box_np[0], ax[1])
            ax[1].set_title("MedSAM Segmentation")
            
            vis_output_path = join(vis_output_folder, f"vis_{os.path.splitext(img_file)[0]}.png")
            fig.savefig(vis_output_path)
            plt.close(fig)
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")

if __name__ == "__main__":
    # Set your paths here
    image_folder = "dataset/CVC-300_images"  # Folder containing original images
    bbox_folder = "dataset/CVC-300_TXT"     # Folder containing bbox txt files
    mask_output_folder = "CVC-300_outputs"       # Folder to save segmentation masks
    vis_output_folder = "CVC-300_visualization"  # Folder to save visualization images
    
    process_folder(image_folder, bbox_folder, mask_output_folder, vis_output_folder)