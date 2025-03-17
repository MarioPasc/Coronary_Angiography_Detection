import os
import imagehash
import cv2
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from skimage.metrics import structural_similarity as ssim

def compute_image_hash(img_path):
    """Compute perceptual hash for a single image."""
    try:
        img = Image.open(img_path)
        # Perceptual hash is good for exact image matching
        img_hash = str(imagehash.phash(img))
        return os.path.basename(img_path), img_hash
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, None

def create_image_hash_dict(image_dir):
    """Create a dictionary mapping filenames to their image hashes."""
    hash_dict = {}
    img_paths = [os.path.join(image_dir, filename) 
                for filename in os.listdir(image_dir) 
                if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Use parallel processing for speed
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(compute_image_hash, img_paths),
            total=len(img_paths),
            desc=f"Processing {os.path.basename(image_dir)}"
        ))
    
    for filename, img_hash in results:
        if filename and img_hash:
            hash_dict[filename] = img_hash
    
    return hash_dict

def match_by_hash(stenosis_dir, syntax_dir):
    """Match images using perceptual hash."""
    if not (os.path.exists(stenosis_dir) and os.path.exists(syntax_dir)):
        print(f"Directories not found")
        return {}, [], []
    
    print(f"\nMatching images by hash...")
    
    # Create hash dictionaries: {filename: hash}
    stenosis_hashes = create_image_hash_dict(stenosis_dir)
    syntax_hashes = create_image_hash_dict(syntax_dir)
    
    print(f"Found {len(stenosis_hashes)} stenosis images and {len(syntax_hashes)} syntax images")
    
    # Create syntax hash to filename mapping for quick lookup
    syntax_hash_to_file = {}
    for filename, hash_val in syntax_hashes.items():
        syntax_hash_to_file[hash_val] = filename
    
    # Match stenosis images to syntax images
    stenosis_to_syntax = {}
    matched_stenosis_files = set()
    matched_syntax_files = set()
    
    for stenosis_filename, hash_val in tqdm(stenosis_hashes.items(), desc="Matching by hash"):
        if hash_val in syntax_hash_to_file:
            syntax_filename = syntax_hash_to_file[hash_val]
            stenosis_to_syntax[stenosis_filename] = syntax_filename
            matched_stenosis_files.add(stenosis_filename)
            matched_syntax_files.add(syntax_filename)
    
    print(f"Found {len(stenosis_to_syntax)} matches by hash")
    
    # Get unmatched files
    unmatched_stenosis = [f for f in stenosis_hashes.keys() if f not in matched_stenosis_files]
    unmatched_syntax = [f for f in syntax_hashes.keys() if f not in matched_syntax_files]
    
    print(f"Remaining unmatched: {len(unmatched_stenosis)} stenosis images and {len(unmatched_syntax)} syntax images")
    
    return stenosis_to_syntax, unmatched_stenosis, unmatched_syntax

def compare_images_difference(stenosis_path, syntax_path, target_size=(256, 256)):
    """Compare two images using pixel-wise difference."""
    try:
        # Read and resize images for comparison
        stenosis_img = cv2.imread(stenosis_path, cv2.IMREAD_GRAYSCALE)
        syntax_img = cv2.imread(syntax_path, cv2.IMREAD_GRAYSCALE)
        
        if stenosis_img is None or syntax_img is None:
            return float('inf')
        
        # Resize both images to the same dimensions
        stenosis_img = cv2.resize(stenosis_img, target_size)
        syntax_img = cv2.resize(syntax_img, target_size)
        
        # Calculate absolute pixel-wise difference
        diff = cv2.absdiff(stenosis_img, syntax_img)
        
        # Calculate mean pixel difference (lower is better)
        mean_diff = np.mean(diff)
        return mean_diff
    except Exception as e:
        print(f"Error comparing images: {e}")
        return float('inf')

def match_by_pixel_difference(stenosis_dir, syntax_dir, unmatched_stenosis, unmatched_syntax, threshold=5.0):
    """Match unmatched images using pixel-wise difference (lower is better)."""
    if not unmatched_stenosis or not unmatched_syntax:
        return {}
    
    print(f"\nMatching remaining {len(unmatched_stenosis)} images by pixel difference...")
    
    stenosis_to_syntax_diff = {}
    matched_syntax = set()  # Keep track of already matched syntax files
    batch_size = min(100, len(unmatched_stenosis))  # Process in batches to show progress
    
    for i in range(0, len(unmatched_stenosis), batch_size):
        batch = unmatched_stenosis[i:i+batch_size]
        
        for stenosis_file in tqdm(batch, desc=f"Pixel diff batch {i//batch_size+1}/{len(unmatched_stenosis)//batch_size+1}"):
            stenosis_path = os.path.join(stenosis_dir, stenosis_file)
            
            best_match = None
            lowest_diff = threshold  # Only consider matches below threshold
            
            for syntax_file in unmatched_syntax:
                # Skip if this syntax file has already been matched
                if syntax_file in matched_syntax:
                    continue
                
                syntax_path = os.path.join(syntax_dir, syntax_file)
                diff = compare_images_difference(stenosis_path, syntax_path)
                
                if diff < lowest_diff:
                    lowest_diff = diff
                    best_match = syntax_file
            
            if best_match:
                stenosis_to_syntax_diff[stenosis_file] = best_match
                matched_syntax.add(best_match)  # Mark this syntax file as matched
    
    print(f"Found {len(stenosis_to_syntax_diff)} additional matches by pixel difference")
    return stenosis_to_syntax_diff

def compare_images_ssim(stenosis_path, syntax_path, target_size=(256, 256)):
    """Compare two images using SSIM."""
    try:
        # Read and resize images for comparison
        stenosis_img = cv2.imread(stenosis_path, cv2.IMREAD_GRAYSCALE)
        syntax_img = cv2.imread(syntax_path, cv2.IMREAD_GRAYSCALE)
        
        if stenosis_img is None or syntax_img is None:
            return -1
        
        stenosis_img = cv2.resize(stenosis_img, target_size)
        syntax_img = cv2.resize(syntax_img, target_size)
        
        # Calculate SSIM
        score = ssim(stenosis_img, syntax_img)
        return score
    except Exception as e:
        print(f"Error comparing images: {e}")
        return -1

def match_by_ssim(stenosis_dir, syntax_dir, unmatched_stenosis, unmatched_syntax, threshold=0.90):
    """Match unmatched images using SSIM."""
    if not unmatched_stenosis or not unmatched_syntax:
        return {}
    
    print(f"\nMatching remaining {len(unmatched_stenosis)} images by SSIM (this may take a while)...")
    
    stenosis_to_syntax_ssim = {}
    matched_syntax = set()  # Keep track of already matched syntax files
    batch_size = min(100, len(unmatched_stenosis))  # Process in batches to show progress
    
    for i in range(0, len(unmatched_stenosis), batch_size):
        batch = unmatched_stenosis[i:i+batch_size]
        
        for stenosis_file in tqdm(batch, desc=f"SSIM batch {i//batch_size+1}/{len(unmatched_stenosis)//batch_size+1}"):
            stenosis_path = os.path.join(stenosis_dir, stenosis_file)
            
            best_match = None
            highest_score = threshold  # Only consider matches above threshold
            
            for syntax_file in unmatched_syntax:
                # Skip if this syntax file has already been matched
                if syntax_file in matched_syntax:
                    continue
                
                syntax_path = os.path.join(syntax_dir, syntax_file)
                score = compare_images_ssim(stenosis_path, syntax_path)
                
                if score > highest_score:
                    highest_score = score
                    best_match = syntax_file
            
            if best_match:
                stenosis_to_syntax_ssim[stenosis_file] = best_match
                matched_syntax.add(best_match)  # Mark this syntax file as matched
    
    print(f"Found {len(stenosis_to_syntax_ssim)} additional matches by SSIM")
    return stenosis_to_syntax_ssim

def match_images(stenosis_dir, syntax_dir, use_pixel_diff=True, pixel_diff_threshold=5.0, 
                use_ssim=True, ssim_threshold=0.90):
    """Match images using hash first, then pixel difference, then SSIM for remaining images."""
    # Phase 1: Match by hash (fast and accurate)
    hash_matches, unmatched_stenosis, unmatched_syntax = match_by_hash(stenosis_dir, syntax_dir)
    
    combined_matches = hash_matches.copy()
    diff_matches_count = 0
    
    # Phase 2: Match remaining by pixel difference (faster than SSIM, better than hash)
    if use_pixel_diff and unmatched_stenosis and unmatched_syntax:
        diff_matches = match_by_pixel_difference(stenosis_dir, syntax_dir, 
                                               unmatched_stenosis, unmatched_syntax, 
                                               threshold=pixel_diff_threshold)
        
        combined_matches.update(diff_matches)
        diff_matches_count = len(diff_matches)
        
        # Update unmatched lists
        unmatched_stenosis = [f for f in unmatched_stenosis if f not in diff_matches]
        unmatched_syntax = [f for f in unmatched_syntax if f not in diff_matches.values()]
    
    ssim_matches_count = 0
    
    # Phase 3: Match remaining by SSIM (slowest but can handle more variations)
    if use_ssim and unmatched_stenosis and unmatched_syntax:
        ssim_matches = match_by_ssim(stenosis_dir, syntax_dir, 
                                    unmatched_stenosis, unmatched_syntax, 
                                    threshold=ssim_threshold)
        
        combined_matches.update(ssim_matches)
        ssim_matches_count = len(ssim_matches)
    
    total_images = len([f for f in os.listdir(stenosis_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"\nTotal matches found: {len(combined_matches)} out of {total_images} images ({len(combined_matches)/total_images*100:.2f}%)")
    
    return {
        "matches": combined_matches,
        "hash_matches": len(hash_matches),
        "pixel_diff_matches": diff_matches_count,
        "ssim_matches": ssim_matches_count,
        "total_matched": len(combined_matches),
        "unmatched": total_images - len(combined_matches)
    }

if __name__ == "__main__":
    # Update these paths to your dataset locations
    stenosis_dir = "/home/mario/Python/Datasets/COMBINED/tasks/stenosis_detection/images"
    syntax_dir = "/home/mario/Python/Datasets/COMBINED/tasks/arteries_segmentation/images"
    
    # Match images using all three techniques in sequence
    result = match_images(
        stenosis_dir, syntax_dir, 
        use_pixel_diff=True, pixel_diff_threshold=5.0,
        use_ssim=True, ssim_threshold=0.90
    )
    
    # Save just the image matches to a separate file for ease of use
    with open('stenosis_to_syntax_mapping.json', 'w') as f:
        json.dump(result["matches"], f, indent=4)
    
    # Save the full result with statistics
    with open('matching_results.json', 'w') as f:
        json.dump(result, f, indent=4)
    
    print("\nMapping complete and saved to 'stenosis_to_syntax_mapping.json'")
    print(f"Hash matches: {result['hash_matches']}")
    print(f"Pixel diff matches: {result['pixel_diff_matches']}")
    print(f"SSIM matches: {result['ssim_matches']}")
    print(f"Total matched: {result['total_matched']}")
    print(f"Unmatched: {result['unmatched']}")
    
    # Print some example mappings
    if result["matches"]:
        print("\nExample mappings:")
        items = list(result["matches"].items())[:5]
        for stenosis_file, syntax_file in items:
            print(f"  Stenosis: {stenosis_file} → Syntax: {syntax_file}")