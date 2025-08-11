import torch
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import sys
from copy import deepcopy

# Add EfficientLoFTR to Python path
sys.path.append('thirdparty/EfficientLoFTR')
from src.loftr import LoFTR, full_default_cfg, reparameter

def visualize_matches(img0, img1, mkpts0, mkpts1, mconf, conf_thresh=0.2, save_path=None, show=True):
    """
    Visualize feature matches between two images
    """
    # Filter matches by confidence
    mask = mconf > conf_thresh
    mkpts0_filtered = mkpts0[mask]
    mkpts1_filtered = mkpts1[mask]
    mconf_filtered = mconf[mask]
    
    # Create side-by-side image
    H0, W0 = img0.shape
    H1, W1 = img1.shape
    H = max(H0, H1)
    W = W0 + W1
    
    # Create combined image
    combined_img = np.zeros((H, W), dtype=np.uint8)
    combined_img[:H0, :W0] = img0
    combined_img[:H1, W0:W0+W1] = img1
    
    # Convert to color for visualization
    combined_img_color = cv2.cvtColor(combined_img, cv2.COLOR_GRAY2RGB)
    
    # Draw matches
    for i in range(len(mkpts0_filtered)):
        pt0 = tuple(map(int, mkpts0_filtered[i]))
        pt1 = tuple(map(int, mkpts1_filtered[i] + np.array([W0, 0])))
        
        # Color based on confidence (red = high, blue = low)
        conf = mconf_filtered[i]
        color = (int(255 * conf), int(255 * (1 - conf)), 0)
        
        # Draw keypoints
        cv2.circle(combined_img_color, pt0, 2, color, -1)
        cv2.circle(combined_img_color, pt1, 2, color, -1)
        
        # Draw lines
        cv2.line(combined_img_color, pt0, pt1, color, 1)
    
    if save_path:
        cv2.imwrite(save_path, combined_img_color)
        print(f"Visualization saved to: {save_path}")
    
    if show:
        plt.figure(figsize=(15, 8))
        plt.imshow(combined_img_color)
        plt.title(f"Feature Matches (Confidence > {conf_thresh}): {len(mkpts0_filtered)} matches")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return combined_img_color

def save_matches(mkpts0, mkpts1, mconf, save_path):
    """
    Save match results to a file
    """
    # Save as numpy arrays
    if save_path.endswith('.npz'):
        np.savez(save_path, 
                 mkpts0=mkpts0, 
                 mkpts1=mkpts1, 
                 mconf=mconf)
    else:
        # Save as text file
        with open(save_path, 'w') as f:
            f.write("# Keypoint matches from EfficientLoFTR\n")
            f.write("# Format: x0 y0 x1 y1 confidence\n")
            for i in range(len(mkpts0)):
                f.write(f"{mkpts0[i][0]:.2f} {mkpts0[i][1]:.2f} "
                       f"{mkpts1[i][0]:.2f} {mkpts1[i][1]:.2f} "
                       f"{mconf[i]:.4f}\n")
    
    print(f"Matches saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='EfficientLoFTR Feature Matching with Visualization')
    parser.add_argument('--img0', type=str, default="Data/image_1.jpg", 
                       help='Path to first image')
    parser.add_argument('--img1', type=str, default="Data/image_2.jpg", 
                       help='Path to second image')
    parser.add_argument('--weights', type=str, default="thirdparty/EfficientLoFTR/weights/eloftr_outdoor.ckpt", 
                       help='Path to model weights')
    parser.add_argument('--output_dir', type=str, default="results", 
                       help='Output directory for results')
    parser.add_argument('--conf_thresh', type=float, default=0.2, 
                       help='Confidence threshold for visualization')
    parser.add_argument('--save_vis', action='store_true', 
                       help='Save visualization image')
    parser.add_argument('--save_matches', action='store_true', 
                       help='Save match results')
    parser.add_argument('--no_show', action='store_true', 
                       help='Do not display visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the matcher with default settings
    _default_cfg = deepcopy(full_default_cfg)
    matcher = LoFTR(config=_default_cfg)
    
    # Load pretrained weights
    matcher.load_state_dict(torch.load(args.weights)['state_dict'])
    matcher = reparameter(matcher)  # Essential for good performance
    matcher = matcher.eval().cuda()
    
    # Load and preprocess images
    img0_raw = cv2.imread(args.img0, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(args.img1, cv2.IMREAD_GRAYSCALE)
    
    if img0_raw is None or img1_raw is None:
        raise FileNotFoundError("Could not load one or both images")
    
    print(f"Loaded images: {img0_raw.shape} and {img1_raw.shape}")
    
    # Resize images to be divisible by 32
    img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))
    img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))
    
    # Convert to tensors
    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    batch = {'image0': img0, 'image1': img1}
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()  # Matched keypoints in image0
        mkpts1 = batch['mkpts1_f'].cpu().numpy()  # Matched keypoints in image1
        mconf = batch['mconf'].cpu().numpy()      # Matching confidence scores
    
    print(f"Found {len(mkpts0)} matches")
    print(f"Average confidence: {np.mean(mconf):.3f}")
    
    # Visualize matches
    vis_save_path = os.path.join(args.output_dir, "matches_visualization.png") if args.save_vis else None
    visualize_matches(img0_raw, img1_raw, mkpts0, mkpts1, mconf, 
                     conf_thresh=args.conf_thresh, 
                     save_path=vis_save_path, 
                     show=not args.no_show)
    
    # Save matches
    if args.save_matches:
        matches_save_path = os.path.join(args.output_dir, "matches.npz")
        save_matches(mkpts0, mkpts1, mconf, matches_save_path)
        
        # Also save as text file
        txt_save_path = os.path.join(args.output_dir, "matches.txt")
        save_matches(mkpts0, mkpts1, mconf, txt_save_path)

if __name__ == "__main__":
    main()