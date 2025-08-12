import torch
import cv2
import numpy as np
import argparse
import time
import sys
from copy import deepcopy

# Add EfficientLoFTR to Python path
sys.path.append('thirdparty/EfficientLoFTR')
from src.loftr import LoFTR, full_default_cfg, reparameter

class RealTimeStereoMatcher:
    def __init__(self, weights_path, conf_thresh=0.2, resize_factor=1.0):
        self.conf_thresh = conf_thresh
        self.resize_factor = resize_factor
        
        # Initialize the matcher
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        _default_cfg = deepcopy(full_default_cfg)
        self.matcher = LoFTR(config=_default_cfg)
        
        # Load pretrained weights
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.matcher.load_state_dict(checkpoint['state_dict'])
        self.matcher = reparameter(self.matcher)
        self.matcher = self.matcher.eval().to(self.device)
        
        print("Model loaded successfully!")
        
        # Performance tracking
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.processing_times = []
        
    def preprocess_image(self, img):
        """Preprocess image for the model"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
            
        # Resize if specified
        if self.resize_factor != 1.0:
            h, w = img_gray.shape
            new_h, new_w = int(h * self.resize_factor), int(w * self.resize_factor)
            img_gray = cv2.resize(img_gray, (new_w, new_h))
        
        # Ensure dimensions are divisible by 32
        h, w = img_gray.shape
        new_h = (h // 32) * 32
        new_w = (w // 32) * 32
        
        if new_h != h or new_w != w:
            img_gray = cv2.resize(img_gray, (new_w, new_h))
        
        return img_gray
    
    def create_visualization(self, img0, img1, mkpts0, mkpts1, mconf):
        """Create side-by-side visualization with matches"""
        # Filter matches by confidence
        mask = mconf > self.conf_thresh
        mkpts0_filtered = mkpts0[mask]
        mkpts1_filtered = mkpts1[mask]
        mconf_filtered = mconf[mask]
        
        H0, W0 = img0.shape
        H1, W1 = img1.shape
        H = max(H0, H1)
        W = W0 + W1
        
        # Create combined image
        combined_img = np.zeros((H, W), dtype=np.uint8)
        combined_img[:H0, :W0] = img0
        combined_img[:H1, W0:W0+W1] = img1
        
        # Convert to color
        combined_img_color = cv2.cvtColor(combined_img, cv2.COLOR_GRAY2BGR)
        
        # Draw matches
        for i in range(len(mkpts0_filtered)):
            pt0 = tuple(map(int, mkpts0_filtered[i]))
            pt1 = tuple(map(int, mkpts1_filtered[i] + np.array([W0, 0])))
            
            # Color based on confidence (green to red)
            conf = mconf_filtered[i]
            color = (0, int(255 * (1 - conf)), int(255 * conf))  # BGR format
            
            # Draw keypoints
            cv2.circle(combined_img_color, pt0, 2, color, -1)
            cv2.circle(combined_img_color, pt1, 2, color, -1)
            
            # Draw lines
            cv2.line(combined_img_color, pt0, pt1, color, 1)
        
        # Add text information
        info_text = f"Matches: {len(mkpts0_filtered)} (conf > {self.conf_thresh:.2f})"
        cv2.putText(combined_img_color, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return combined_img_color, len(mkpts0_filtered)
    
    def create_original_view(self, img0, img1):
        """Create side-by-side view of original camera feeds"""
        H0, W0 = img0.shape[:2]
        H1, W1 = img1.shape[:2]
        H = max(H0, H1)
        W = W0 + W1
        
        # Create combined image
        combined_img = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Convert grayscale to color if needed
        if len(img0.shape) == 2:
            img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        else:
            img0_color = img0
            
        if len(img1.shape) == 2:
            img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        else:
            img1_color = img1
        
        combined_img[:H0, :W0] = img0_color
        combined_img[:H1, W0:W0+W1] = img1_color
        
        # Add labels
        cv2.putText(combined_img, "Camera 0", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_img, "Camera 1", (W0 + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return combined_img
    
    def run_matching(self, img0_gray, img1_gray):
        """Run feature matching on preprocessed images"""
        start_time = time.time()
        
        # Convert to tensors
        img0_tensor = torch.from_numpy(img0_gray)[None][None].to(self.device).float() / 255.0
        img1_tensor = torch.from_numpy(img1_gray)[None][None].to(self.device).float() / 255.0
        batch = {'image0': img0_tensor, 'image1': img1_tensor}
        
        # Inference
        with torch.no_grad():
            self.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return mkpts0, mkpts1, mconf, processing_time
    
    def run(self, cam0_id=0, cam1_id=2, skip_frames=2):
        """Main execution loop"""
        # Initialize cameras
        cap0 = cv2.VideoCapture(cam0_id)
        cap1 = cv2.VideoCapture(cam1_id)
        
        # Set camera properties for better performance
        cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap0.isOpened() or not cap1.isOpened():
            print("Error: Could not open cameras")
            return
        
        print("Cameras initialized. Press 'q' to quit, 'r' to reset stats, 's' to save frame")
        
        frame_skip_counter = 0
        
        while True:
            # Capture frames
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            
            if not ret0 or not ret1:
                print("Error: Could not read from cameras")
                break
            
            self.frame_count += 1
            
            # Skip frames for performance
            if frame_skip_counter < skip_frames:
                frame_skip_counter += 1
                # Still show original feeds
                original_view = self.create_original_view(frame0, frame1)
                cv2.imshow('Original Camera Feeds', original_view)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            frame_skip_counter = 0
            
            # Preprocess images
            img0_gray = self.preprocess_image(frame0)
            img1_gray = self.preprocess_image(frame1)
            
            try:
                # Run matching
                mkpts0, mkpts1, mconf, proc_time = self.run_matching(img0_gray, img1_gray)
                
                # Create visualizations
                match_vis, num_matches = self.create_visualization(img0_gray, img1_gray, mkpts0, mkpts1, mconf)
                original_view = self.create_original_view(frame0, frame1)
                
                # Calculate and display performance stats
                current_time = time.time()
                fps = self.frame_count / (current_time - self.fps_start_time)
                avg_proc_time = np.mean(self.processing_times[-30:])  # Last 30 frames
                
                # Add performance info to match visualization
                perf_text = f"FPS: {fps:.1f} | Proc: {proc_time*1000:.1f}ms | Avg: {avg_proc_time*1000:.1f}ms"
                cv2.putText(match_vis, perf_text, (10, match_vis.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display windows
                cv2.imshow('Original Camera Feeds', original_view)
                cv2.imshow('Feature Matches', match_vis)
                
                print(f"Frame {self.frame_count}: {num_matches} matches, {proc_time*1000:.1f}ms processing")
                
            except Exception as e:
                print(f"Error in matching: {e}")
                # Still show original feeds
                original_view = self.create_original_view(frame0, frame1)
                cv2.imshow('Original Camera Feeds', original_view)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset statistics
                self.frame_count = 0
                self.fps_start_time = time.time()
                self.processing_times = []
                print("Statistics reset")
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                if 'match_vis' in locals():
                    cv2.imwrite(f'match_frame_{timestamp}.png', match_vis)
                    print(f"Saved match_frame_{timestamp}.png")
        
        # Cleanup
        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        if self.processing_times:
            print(f"\nFinal Statistics:")
            print(f"Total frames: {self.frame_count}")
            print(f"Average processing time: {np.mean(self.processing_times)*1000:.1f}ms")
            print(f"Min processing time: {np.min(self.processing_times)*1000:.1f}ms")
            print(f"Max processing time: {np.max(self.processing_times)*1000:.1f}ms")

def main():
    parser = argparse.ArgumentParser(description='Real-time Stereo Feature Matching')
    parser.add_argument('--weights', type=str, 
                       default="thirdparty/EfficientLoFTR/weights/eloftr_outdoor.ckpt", 
                       help='Path to model weights')
    parser.add_argument('--cam0', type=int, default=0, help='Camera 0 ID')
    parser.add_argument('--cam1', type=int, default=1, help='Camera 1 ID')
    parser.add_argument('--conf_thresh', type=float, default=0.2, 
                       help='Confidence threshold for visualization')
    parser.add_argument('--resize_factor', type=float, default=1.0, 
                       help='Resize factor for input images (for performance)')
    parser.add_argument('--skip_frames', type=int, default=2, 
                       help='Number of frames to skip between processing')
    
    args = parser.parse_args()
    
    try:
        matcher = RealTimeStereoMatcher(
            weights_path=args.weights,
            conf_thresh=args.conf_thresh,
            resize_factor=args.resize_factor
        )
        
        matcher.run(args.cam0, args.cam1, args.skip_frames)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Two cameras connected (or use --cam0 and --cam1 to specify camera IDs)")
        print("2. The model weights file exists")
        print("3. CUDA drivers installed (if using GPU)")

if __name__ == "__main__":
    main()