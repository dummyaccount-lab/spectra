#!/usr/bin/env python3
"""
Depth Map Densification using Semantic Segmentation

This module provides functionality to densify sparse depth maps using semantic segmentation
information. It implements various algorithms including KNN-based filling, occlusion filtering,
and semantic-aware smoothing techniques.

Author: Depth Map Densification Team
Date: August 2025
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial import cKDTree
from scipy.ndimage import minimum_filter, gaussian_filter
import os
from typing import Tuple, Dict, List, Optional, Union


class SemanticColormap:
    """Handles semantic segmentation colormap operations for Cityscapes dataset."""
    
    def __init__(self):
        self.colormap = self._create_label_colormap()
        self.reduced_label_mapping = self._create_reduced_mapping()
        self.reduced_colormap = self._create_reduced_colormap()
        self.label_names = self._get_label_names()
        self.reduced_label_names = self._get_reduced_label_names()
        self.dynamic_class_ids = self._get_dynamic_class_ids()
    
    def _create_label_colormap(self) -> np.ndarray:
        """Creates a label colormap used in CITYSCAPES segmentation benchmark."""
        palette = [
            [165,42,42],[0,192,0],[196,196,196],[190,153,153],[180,165,180],
            [90,120,150],[102,102,156],[128,64,255],[140,140,200],[170,170,170],
            [250,170,160],[96,96,96],[230,150,140],[128,64,128],[110,110,110],
            [244,35,232],[150,100,100],[70,70,70],[150,120,90],[220,20,60],
            [255,0,0],[255,0,100],[255,0,200],[200,128,128],[255,255,255],
            [64,170,64],[230,160,50],[70,130,180],[190,255,255],[152,251,152],
            [107,142,35],[0,170,30],[255,255,128],[250,0,30],[100,140,180],
            [220,220,220],[220,128,128],[222,40,40],[100,170,30],[40,40,40],
            [33,33,33],[100,128,160],[142,0,0],[70,100,150],[210,170,100],
            [153,153,153],[128,128,128],[0,0,80],[250,170,30],[192,192,192],
            [220,220,0],[140,140,20],[119,11,32],[150,0,255],[0,60,100],
            [0,0,142],[0,0,90],[0,0,230],[0,80,100],[128,64,64],[0,0,110],
            [0,0,70],[0,0,192],[32,32,32],[120,10,10],
        ]
        
        colormap = np.zeros((256, 3), dtype=np.uint8)
        for i, color in enumerate(palette):
            colormap[i] = color
        return colormap
    
    def _get_label_names(self) -> np.ndarray:
        """Returns the original 66 class names."""
        return np.asarray([
            'Bird','Ground Animal','Curb','Fence','Guard Rail','Barrier','Wall',
            'Bike Lane','Crosswalk - Plain','Curb Cut','Parking','Pedestrian Area',
            'Rail Track','Road','Service Lane','Sidewalk','Bridge','Building','Tunnel',
            'Person','Bicyclist','Motorcyclist','Other Rider','Lane Marking - Crosswalk',
            'Lane Marking - General','Mountain','Sand','Sky','Snow','Terrain',
            'Vegetation','Water','Banner','Bench','Bike Rack','Billboard','Catch Basin',
            'CCTV Camera','Fire Hydrant','Junction Box','Mailbox','Manhole','Phone Booth',
            'Pothole','Street Light','Pole','Traffic Sign Frame','Utility Pole',
            'Traffic Light','Traffic Sign (Back)','Traffic Sign (Front)','Trash Can',
            'Bicycle','Boat','Bus','Car','Caravan','Motorcycle','On Rails',
            'Other Vehicle','Trailer','Truck','Wheeled Slow','Car Mount','Ego Vehicle',
        ])
    
    def _create_reduced_mapping(self) -> np.ndarray:
        """Maps 66-class indices to 15 reduced-class indices."""
        reduced_label_mapping = np.zeros(256, dtype=np.uint8)
        
        groups = {
            1: [13, 14, 15, 7, 8, 9, 10, 11, 12, 24],              # Flat
            2: [3, 4, 5, 6, 16, 17, 18],                           # Construction
            3: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 48],       # Object
            4: [30],                                               # Vegetation
            5: [27],                                               # Sky
            6: [49, 50, 51, 52, 53, 54, 55, 56, 57, 58],           # Vehicle
            7: [19, 20, 21, 22],                                   # Dynamic (people/riders)
            8: [31],                                               # Water
            9: [28, 29],                                           # Terrain (snow, terrain)
            10: [45, 46, 47],                                      # Signage
            11: [33, 34],                                          # Furniture
            12: [2],                                               # Barrier
            13: [0, 1],                                            # Animal
            14: [59],                                              # Ego Vehicle
            0: [23, 25, 26, 32, 60, 61, 62, 63, 64, 65],           # Void & undefined
        }
        
        for class_id, indices in groups.items():
            for idx in indices:
                reduced_label_mapping[idx] = class_id
        
        return reduced_label_mapping
    
    def _create_reduced_colormap(self) -> np.ndarray:
        """Creates colormap for reduced classes."""
        num_reduced_classes = self.reduced_label_mapping.max() + 1
        reduced_colormap = np.zeros((num_reduced_classes, 3), dtype=np.uint8)
        
        for reduced_class in range(num_reduced_classes):
            orig_class_indices = np.where(self.reduced_label_mapping == reduced_class)[0]
            if len(orig_class_indices) == 0:
                reduced_colormap[reduced_class] = [0, 0, 0]
            else:
                reduced_colormap[reduced_class] = self.colormap[orig_class_indices[0]]
        
        return reduced_colormap
    
    def _get_reduced_label_names(self) -> np.ndarray:
        """Returns reduced class names."""
        reduced_label_names_map = {
            0: 'Void & Undefined',
            1: 'Flat',
            2: 'Construction',
            3: 'Object',
            4: 'Vegetation',
            5: 'Sky',
            6: 'Vehicle',
            7: 'Dynamic (People/Riders)',
            8: 'Water',
            9: 'Terrain',
            10: 'Signage',
            11: 'Furniture',
            12: 'Barrier',
            13: 'Animal',
            14: 'Ego Vehicle',
        }
        
        num_reduced_classes = max(reduced_label_names_map.keys()) + 1
        label_names_reduced = [reduced_label_names_map.get(i, 'Unknown') for i in range(num_reduced_classes)]
        return np.array(label_names_reduced)
    
    def _get_dynamic_class_ids(self) -> List[int]:
        """Returns indices of dynamic classes for special processing."""
        dynamic_class_names = ['Vehicle', 'Dynamic (People/Riders)', 'Signage', 'Object', 'Ego Vehicle']
        return [i for i, name in enumerate(self.reduced_label_names) if name in dynamic_class_names]


class DepthProcessor:
    """Main class for depth map densification operations."""
    
    def __init__(self, semantic_colormap: SemanticColormap):
        self.colormap_handler = semantic_colormap
        self.camera_params = {
            'baseline': 0.4662,  # meters
            'focal': 1680,       # pixels
            'intrinsics': np.array([
                [1685.65008, 0, 629.89747],
                [0, 1686.0903, 373.90386],
                [0, 0, 1]
            ])
        }
    
    def decode_segmentation_map(self, instance_image: np.ndarray, threshold: int = 80) -> np.ndarray:
        """
        Decode RGB segmentation image back to reduced class IDs.
        
        Args:
            instance_image: RGB image (H, W, 3), uint8
            threshold: Color matching threshold
            
        Returns:
            segmentation_map_reduced: 2D array (H, W) of reduced class IDs
        """
        height, width = instance_image.shape[:2]
        segmentation_map = np.full((height, width), 255, dtype=np.uint8)
        
        pixels = instance_image.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        
        for color in unique_colors:
            distances = np.linalg.norm(
                self.colormap_handler.colormap.astype(np.int16) - color.astype(np.int16), 
                axis=1
            )
            closest_idx = np.argmin(distances)
            
            if distances[closest_idx] <= threshold:
                class_id = closest_idx
            else:
                class_id = 255  # Unknown
            
            mask = np.all(instance_image == color.reshape(1, 1, 3), axis=2)
            segmentation_map[mask] = class_id
            
            if class_id == 255:
                print(f"Unknown color {tuple(color)} ignored (closest distance {distances[closest_idx]:.1f})")
        
        return self.colormap_handler.reduced_label_mapping[segmentation_map]
    
    def reverse_inferno_colormap(self, depth_color_image_rgb: np.ndarray,
                                explicit_missing_marker_rgb: Optional[List[int]] = None,
                                no_lidar_return_bkg_rgb: List[int] = [252, 255, 164],
                                color_match_threshold_sq: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts an inferno-colored RGB depth image back to depth and disparity maps.
        
        Args:
            depth_color_image_rgb: RGB depth image with inferno colormap
            explicit_missing_marker_rgb: RGB color for explicit missing markers
            no_lidar_return_bkg_rgb: RGB color for no lidar return background
            color_match_threshold_sq: Threshold for color matching
            
        Returns:
            Tuple of (depth_norm, disparity_map)
        """
        img = depth_color_image_rgb.astype(np.uint8)
        H, W, _ = img.shape
        
        # Build the inferno colormap reference
        cmap = (plt.cm.inferno(np.arange(256))[:, :3] * 255).astype(np.uint8)
        tree = cKDTree(cmap.astype(np.int32))
        
        pixels = img.reshape(-1, 3).astype(np.int32)
        dists_sq, idxs = tree.query(pixels, k=1)
        
        depth_norm = np.full((H * W,), np.nan, dtype=np.float32)
        
        # Handle missing markers
        mask_missing = np.zeros(H * W, bool)
        if explicit_missing_marker_rgb is not None:
            missing_color = np.array(explicit_missing_marker_rgb, dtype=np.int32)
            mask_missing = np.all(pixels == missing_color, axis=1)
        
        # Handle no-lidar return background
        mask_noreturn = np.zeros(H * W, bool)
        if no_lidar_return_bkg_rgb is not None:
            no_return_color = np.array(no_lidar_return_bkg_rgb, dtype=np.int32)
            mask_noreturn = np.all(pixels == no_return_color, axis=1)
        
        # Assign normalized depth for good matches
        valid = ~(mask_missing | mask_noreturn)
        good_match = (dists_sq <= color_match_threshold_sq) & valid
        depth_norm[good_match] = idxs[good_match].astype(np.float32) / 255.0
        
        depth_norm = depth_norm.reshape(H, W)
        
        # Compute disparity safely
        with np.errstate(divide='ignore', invalid='ignore'):
            disparity_map = (self.camera_params['baseline'] * self.camera_params['focal']) / depth_norm
            disparity_map[~np.isfinite(disparity_map)] = np.nan
        
        return depth_norm, disparity_map
    
    def resize_segmentation(self, instance_image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        """
        Resize segmentation image while preserving class labels.
        
        Args:
            instance_image: RGB segmentation image
            target_height: Target height
            target_width: Target width
            
        Returns:
            Resized RGB segmentation image
        """
        resized = cv2.resize(
            instance_image,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST
        )
        print(f"Resized instance image shape: {resized.shape}")
        return resized


class DepthFiller:
    """Handles various depth filling algorithms."""
    
    def __init__(self, colormap_handler: SemanticColormap):
        self.colormap_handler = colormap_handler
    
    def knn_fill_missing_depths(self, depth_map: np.ndarray, segmentation_map: np.ndarray, 
                               k: int = 25, visualize: bool = False) -> np.ndarray:
        """
        Fill missing depth values using K-Nearest Neighbors based on segmentation classes.
        
        Args:
            depth_map: 2D depth map with NaN for missing values
            segmentation_map: 2D segmentation map with class IDs
            k: Number of nearest neighbors to consider
            visualize: Whether to show overlay visualization
            
        Returns:
            Filled depth map
        """
        if visualize:
            self._visualize_overlay(depth_map, segmentation_map)
        
        filled_depth = depth_map.copy()
        height, width = depth_map.shape
        
        # Identify valid and missing pixels
        valid_mask = ~np.isnan(depth_map)
        missing_mask = np.isnan(depth_map)
        
        if not np.any(valid_mask):
            print("KNN Warning: No valid pixels found in the depth map.")
            return filled_depth
        
        if not np.any(missing_mask):
            print("KNN Info: No missing pixels to fill.")
            return filled_depth
        
        valid_positions_all = np.array(np.where(valid_mask)).T
        valid_depths_all = depth_map[valid_mask]
        
        # Create regressors for each class
        regressors_by_class = {}
        unique_classes_with_valid_data = np.unique(segmentation_map[valid_mask])
        
        for class_id in unique_classes_with_valid_data:
            class_points_mask_on_valid = (segmentation_map[valid_mask] == class_id)
            current_class_positions = valid_positions_all[class_points_mask_on_valid]
            current_class_depths = valid_depths_all[class_points_mask_on_valid]
            
            if len(current_class_positions) >= 25:
                actual_k_class = min(k, len(current_class_positions))
                knn = KNeighborsRegressor(n_neighbors=actual_k_class, weights='uniform')
                knn.fit(current_class_positions, current_class_depths)
                regressors_by_class[class_id] = knn
        
        # Create fallback regressor
        knn_fallback = None
        actual_k_fallback = min(k, len(valid_positions_all))
        if actual_k_fallback > 0:
            knn_fallback = KNeighborsRegressor(n_neighbors=actual_k_fallback, weights='uniform')
            knn_fallback.fit(valid_positions_all, valid_depths_all)
        
        # Fill missing pixels
        unique_classes_in_missing = np.unique(segmentation_map[missing_mask])
        
        for class_id_of_missing in unique_classes_in_missing:
            current_class_missing_mask = np.logical_and(missing_mask, segmentation_map == class_id_of_missing)
            positions_to_predict = np.array(np.where(current_class_missing_mask)).T
            
            if len(positions_to_predict) == 0:
                continue
            
            regressor = regressors_by_class.get(class_id_of_missing)
            
            if regressor:
                predicted_depths = regressor.predict(positions_to_predict)
                for idx, (r, c) in enumerate(positions_to_predict):
                    filled_depth[r, c] = predicted_depths[idx]
            elif knn_fallback:
                predicted_depths = knn_fallback.predict(positions_to_predict)
                for idx, (r, c) in enumerate(positions_to_predict):
                    filled_depth[r, c] = predicted_depths[idx]
        
        return filled_depth
    
    def _visualize_overlay(self, depth_map: np.ndarray, segmentation_map: np.ndarray):
        """Visualize overlay of depth and segmentation maps."""
        depth_norm = np.nan_to_num(depth_map)
        depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min() + 1e-6)
        depth_rgb = (np.stack([depth_norm] * 3, axis=-1) * 255).astype(np.uint8)
        
        color_seg_bgr = cv2.applyColorMap((segmentation_map * 20).astype(np.uint8), cv2.COLORMAP_JET)
        color_seg_rgb = cv2.cvtColor(color_seg_bgr, cv2.COLOR_BGR2RGB)
        
        alpha = 0.2
        overlay = cv2.addWeighted(depth_rgb, 1 - alpha, color_seg_rgb, alpha, 0)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(overlay)
        plt.title("Overlay of Semantic Map and Depth Map")
        plt.axis("off")
        plt.show()
    
    def occlusion_filter(self, depth_map: np.ndarray, segmentation_map: np.ndarray,
                        foreground_classes: List[int], near_threshold: float = 50.0,
                        near_params: Dict = None, far_params: Dict = None,
                        visualize: bool = True) -> np.ndarray:
        """
        Occlusion filtering with two depth-based zones.
        
        Args:
            depth_map: Input depth map with NaNs for invalid pixels
            segmentation_map: Segmentation map with class IDs
            foreground_classes: List of class IDs to process
            near_threshold: Depth cutoff between near/far zones
            near_params: Parameters for near zone filtering
            far_params: Parameters for far zone filtering
            visualize: Whether to show before/after visualization
            
        Returns:
            Filtered depth map
        """
        if near_params is None:
            near_params = {"window_size": 55, "max_depth_diff": 1}
        if far_params is None:
            far_params = {"window_size": 5, "max_depth_diff": 0.1}
        
        filtered_depth = depth_map.copy().astype(np.float32)
        
        for class_id in np.unique(segmentation_map):
            if class_id not in foreground_classes:
                continue
            
            class_mask = (segmentation_map == class_id) & (~np.isnan(depth_map))
            if not np.any(class_mask):
                continue
            
            near_mask = class_mask & (depth_map < near_threshold)
            far_mask = class_mask & (depth_map >= near_threshold)
            
            # Process near region
            if np.any(near_mask):
                masked_depth_near = np.where(near_mask, depth_map, np.inf)
                local_min_near = minimum_filter(
                    masked_depth_near,
                    size=near_params["window_size"],
                    mode='constant',
                    cval=np.inf
                )
                too_far_near = (depth_map - local_min_near > near_params["max_depth_diff"]) & near_mask
                filtered_depth[too_far_near] = np.nan
                print(f"Class {class_id} (near): removed {np.sum(too_far_near)} points")
            
            # Process far region
            if np.any(far_mask):
                masked_depth_far = np.where(far_mask, depth_map, np.inf)
                local_min_far = minimum_filter(
                    masked_depth_far,
                    size=far_params["window_size"],
                    mode='constant',
                    cval=np.inf
                )
                too_far_far = (depth_map - local_min_far > far_params["max_depth_diff"]) & far_mask
                filtered_depth[too_far_far] = np.nan
                print(f"Class {class_id} (far): removed {np.sum(too_far_far)} points")
        
        if visualize:
            self._visualize_filter_results(depth_map, filtered_depth)
        
        return filtered_depth
    
    def _visualize_filter_results(self, original: np.ndarray, filtered: np.ndarray):
        """Visualize filtering results."""
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Depth")
        plt.imshow(original, cmap='inferno_r')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("Filtered Depth (dual-zone)")
        plt.imshow(filtered, cmap='inferno_r')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def semantic_zscore_outlier_filter(self, depth_map: np.ndarray, segmentation_map: np.ndarray,
                                      target_classes: List[int], z_thresh: float = 2) -> np.ndarray:
        """Remove outliers using z-score filtering within semantic classes."""
        filtered = depth_map.copy()
        
        for class_id in np.unique(segmentation_map):
            if class_id not in target_classes:
                continue
            
            class_mask = (segmentation_map == class_id) & (~np.isnan(depth_map))
            if np.sum(class_mask) < 10:
                continue
            
            local_depth = depth_map[class_mask]
            mean = np.mean(local_depth)
            std = np.std(local_depth)
            z = np.abs((local_depth - mean) / (std + 1e-6))
            outlier_mask = (z > z_thresh)
            
            coords = np.argwhere(class_mask)
            for idx, is_outlier in zip(coords, outlier_mask):
                if is_outlier:
                    filtered[tuple(idx)] = np.nan
            
            print(f"Class {class_id}: removed {np.sum(outlier_mask)} outliers")
        
        return filtered
    
    def fill_depth_with_semantics(self, depth_map: np.ndarray, target_classes: List[int],
                                 segmentation_map: Optional[np.ndarray] = None,
                                 densify_iters: int = 3, k: int = 5,
                                 outlier_removal: bool = False) -> np.ndarray:
        """
        Fill missing depth using spatial interpolation with optional semantic guidance.
        
        Args:
            depth_map: Input depth map with NaNs for missing values
            target_classes: List of semantic classes to process
            segmentation_map: Optional segmentation map for class-aware processing
            densify_iters: Number of densification iterations
            k: Number of nearest neighbors for interpolation
            outlier_removal: Whether to remove outliers before filling
            
        Returns:
            Filled depth map
        """
        filled_depth = depth_map.copy().astype(np.float32)
        
        # Step 1: Remove outliers if requested
        if outlier_removal and segmentation_map is not None:
            filled_depth = self.semantic_zscore_outlier_filter(filled_depth, segmentation_map, target_classes)
        
        # Step 2: Interpolation
        if segmentation_map is None:
            # Global fill
            for _ in range(densify_iters):
                valid_mask = ~np.isnan(filled_depth)
                missing_mask = np.isnan(filled_depth)
                
                if np.count_nonzero(valid_mask) < 5 or np.count_nonzero(missing_mask) == 0:
                    break
                
                valid_positions = np.argwhere(valid_mask)
                valid_depths = filled_depth[valid_mask]
                missing_positions = np.argwhere(missing_mask)
                
                tree = cKDTree(valid_positions)
                dists, idxs = tree.query(missing_positions, k=min(k, len(valid_positions)))
                eps = 1e-6
                weights = 1.0 / (dists + eps)
                weights /= weights.sum(axis=1, keepdims=True)
                interp_values = np.sum(weights * valid_depths[idxs], axis=1)
                
                for i, (y, x) in enumerate(missing_positions):
                    filled_depth[y, x] = interp_values[i]
        else:
            # Class-aware local interpolation
            unique_classes = np.unique(segmentation_map)
            
            for _ in range(densify_iters):
                for class_id in unique_classes:
                    class_mask = (segmentation_map == class_id)
                    valid_mask = (~np.isnan(filled_depth)) & class_mask
                    missing_mask = np.isnan(filled_depth) & class_mask
                    
                    if np.count_nonzero(valid_mask) < 5 or np.count_nonzero(missing_mask) == 0:
                        continue
                    
                    valid_positions = np.argwhere(valid_mask)
                    valid_depths = filled_depth[valid_mask]
                    missing_positions = np.argwhere(missing_mask)
                    
                    tree = cKDTree(valid_positions)
                    dists, idxs = tree.query(missing_positions, k=min(k, len(valid_positions)))
                    eps = 1e-6
                    weights = 1.0 / (dists + eps)
                    weights /= weights.sum(axis=1, keepdims=True)
                    interp_values = np.sum(weights * valid_depths[idxs], axis=1)
                    
                    for i, (y, x) in enumerate(missing_positions):
                        filled_depth[y, x] = interp_values[i]
        
        return filled_depth
    
    def semantic_joint_fill_depth(self, depth_map: np.ndarray, segmentation_map: np.ndarray,
                                 foreground_class_ids: List[int], visualize: bool = True) -> np.ndarray:
        """
        Complete pipeline: occlusion filtering + semantic densification.
        
        Args:
            depth_map: Input depth map
            segmentation_map: Segmentation map
            foreground_class_ids: List of semantic classes to process
            visualize: Whether to show intermediate results
            
        Returns:
            Filled and filtered depth map
        """
        depth_map = depth_map.astype(np.float32)
        depth_map[depth_map == 0] = np.nan
        
        # Step 1: Occlusion filtering
        depth_filtered = self.occlusion_filter(
            depth_map,
            segmentation_map,
            foreground_classes=foreground_class_ids,
            visualize=visualize
        )
        
        # Step 2: Semantic densification
        filled_depth = self.fill_depth_with_semantics(
            depth_filtered,
            foreground_class_ids,
            segmentation_map,
            densify_iters=3,
            k=15
        )
        
        return filled_depth


class DepthSmoother:
    """Handles depth map smoothing operations."""
    
    def __init__(self, colormap_handler: SemanticColormap):
        self.colormap_handler = colormap_handler
    
    def smooth_depth_with_guided_segmentation(self, filled_depth: np.ndarray, 
                                            segmentation_map: np.ndarray,
                                            sigma_spatial: int = 5, 
                                            sigma_depth: float = 1.0) -> np.ndarray:
        """
        Smooth depth using segmentation guidance without creating visible borders.
        
        Args:
            filled_depth: Input depth map with missing values filled
            segmentation_map: Segmentation map with class labels
            sigma_spatial: Gaussian smoothing kernel size
            sigma_depth: Weight for penalizing depth differences
            
        Returns:
            Smoothed depth map
        """
        smoothed_depth = np.zeros_like(filled_depth)
        confidence = np.zeros_like(filled_depth)
        
        unique_classes = np.unique(segmentation_map)
        
        for class_id in unique_classes:
            if class_id == 255:  # ignore void
                continue
            
            class_mask = (segmentation_map == class_id).astype(np.float32)
            depth_masked = filled_depth * class_mask
            
            # Use Gaussian filtering within the class
            blurred = gaussian_filter(depth_masked, sigma=sigma_spatial)
            blurred_mask = gaussian_filter(class_mask, sigma=sigma_spatial)
            
            # Avoid division by zero
            class_depth = np.divide(blurred, blurred_mask + 1e-5)
            
            # Combine depth values
            smoothed_depth += class_depth * class_mask
            confidence += class_mask
        
        # Normalize by confidence
        smoothed_depth = np.divide(smoothed_depth, confidence + 1e-5)
        return smoothed_depth
    
    def smooth_depth_connected_objects(self, filled_depth: np.ndarray, 
                                     segmentation_map: np.ndarray, 
                                     radius: int = 50) -> np.ndarray:
        """
        Smooth depth map by connected components within each class.
        
        Args:
            filled_depth: Input filled depth map
            segmentation_map: Segmentation map with class IDs
            radius: Radius for local smoothing window
            
        Returns:
            Smoothed depth map
        """
        depth_smooth = filled_depth.copy()
        height, width = filled_depth.shape
        
        for class_id in self.colormap_handler.dynamic_class_ids:
            class_mask = (segmentation_map == class_id)
            
            # Find connected components in this class
            num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
                class_mask.astype(np.uint8), connectivity=4, ltype=cv2.CV_32S
            )
            
            for label_idx in range(1, num_labels):  # 0 = background
                obj_mask = (labels_map == label_idx)
                if np.sum(obj_mask) == 0:
                    continue
                
                ys, xs = np.where(obj_mask)
                y_min = max(0, ys.min() - radius)
                y_max = min(height, ys.max() + radius + 1)
                x_min = max(0, xs.min() - radius)
                x_max = min(width, xs.max() + radius + 1)
                
                sub_depth = filled_depth[y_min:y_max, x_min:x_max]
                sub_obj_mask = obj_mask[y_min:y_max, x_min:x_max]
                
                # Apply local smoothing for each pixel in the object
                for y, x in zip(ys, xs):
                    yy = y - y_min
                    xx = x - x_min
                    win_y_min = max(0, yy - radius)
                    win_y_max = min(sub_depth.shape[0], yy + radius + 1)
                    win_x_min = max(0, xx - radius)
                    win_x_max = min(sub_depth.shape[1], xx + radius + 1)
                    
                    # Local window within the object
                    local_mask = sub_obj_mask[win_y_min:win_y_max, win_x_min:win_x_max]
                    local_depth = sub_depth[win_y_min:win_y_max, win_x_min:win_x_max][local_mask]
                    local_depth = local_depth[~np.isnan(local_depth)]
                    
                    if len(local_depth) > 0:
                        depth_smooth[y, x] = np.mean(local_depth)
        
        return depth_smooth


class DepthAnalyzer:
    """Handles depth map analysis and visualization."""
    
    def __init__(self, colormap_handler: SemanticColormap):
        self.colormap_handler = colormap_handler
    
    def analyze_depth_by_class(self, depth_map: np.ndarray, segmentation_map: np.ndarray) -> Dict:
        """
        Analyze depth statistics by semantic class.
        
        Args:
            depth_map: 2D depth map
            segmentation_map: 2D segmentation map with class IDs
            
        Returns:
            Dictionary with class statistics
        """
        stats = {}
        unique_classes = np.unique(segmentation_map)
        
        for class_id in unique_classes:
            if class_id < len(self.colormap_handler.label_names):
                class_name = self.colormap_handler.label_names[class_id]
            else:
                class_name = f"Unknown-{class_id}"
            
            class_mask = (segmentation_map == class_id)
            valid_depths = depth_map[np.logical_and(class_mask, ~np.isnan(depth_map))]
            
            if len(valid_depths) > 0:
                stats[class_name] = {
                    'mean': float(np.mean(valid_depths)),
                    'median': float(np.median(valid_depths)),
                    'min': float(np.min(valid_depths)),
                    'max': float(np.max(valid_depths)),
                    'std': float(np.std(valid_depths)),
                    'count': int(len(valid_depths)),
                    'missing': int(np.sum(np.logical_and(class_mask, np.isnan(depth_map))))
                }
            else:
                stats[class_name] = {
                    'mean': None, 'median': None, 'min': None, 'max': None, 'std': None,
                    'count': 0, 'missing': int(np.sum(class_mask))
                }
        
        return stats
    
    def visualize_results(self, original_image: np.ndarray, original_depth: np.ndarray,
                         filled_depth: np.ndarray, segmentation_map: np.ndarray,
                         output_dir: str = "output") -> None:
        """
        Visualize processing results.
        
        Args:
            original_image: Original RGB image
            original_depth: Original depth map
            filled_depth: Filled depth map
            segmentation_map: Segmentation map
            output_dir: Output directory for saving results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert segmentation to RGB using colormap
        seg_viz = self.colormap_handler.reduced_colormap[segmentation_map.astype(np.int32)]
        
        # Normalize depth maps
        def normalize_depth(depth):
            depth = np.nan_to_num(depth, nan=0.0)
            min_val, max_val = depth.min(), depth.max()
            if max_val - min_val > 0:
                return (depth - min_val) / (max_val - min_val)
            else:
                return depth * 0
        
        original_depth_norm = normalize_depth(original_depth)
        filled_depth_norm = normalize_depth(filled_depth)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        
        axes[0, 0].imshow(original_depth_norm, cmap='inferno')
        axes[0, 0].set_title('Original Depth (yellow = missing)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(filled_depth_norm, cmap='inferno')
        axes[0, 1].set_title('Filled Depth')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(seg_viz)
        axes[1, 0].set_title('Segmentation')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(original_image)
        axes[1, 1].set_title('Original Image')
        axes[1, 1].axis('off')
        
        # Add legend
        handles = []
        for i, label in enumerate(self.colormap_handler.reduced_label_names):
            color_patch = plt.Rectangle((0, 0), 1, 1, color=self.colormap_handler.reduced_colormap[i] / 255.0)
            handles.append((color_patch, label))
        
        fig.subplots_adjust(bottom=0.2)
        legend_ax = fig.add_axes([0.1, 0.01, 0.8, 0.15])
        legend_ax.axis('off')
        
        legend_elements = [patch for patch, _ in handles]
        legend_labels = [label for _, label in handles]
        legend_ax.legend(legend_elements, legend_labels, loc='center', ncol=8, frameon=False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_depth_histogram(self, depth_map: np.ndarray, title: str = "Depth Histogram") -> None:
        """Plot histogram of depth values."""
        plt.figure(figsize=(10, 6))
        valid_depths = depth_map[~np.isnan(depth_map)].flatten()
        plt.hist(valid_depths, bins=50, color='blue', alpha=0.7)
        plt.title(title)
        plt.xlabel('Depth Value')
        plt.ylabel('Frequency')
        plt.grid()
        plt.show()
    
    def plot_depth_comparison(self, original_depth: np.ndarray, filled_depth: np.ndarray,
                            title: str = "Depth Maps Comparison", percentile: int = 95) -> None:
        """Plot original and filled depth maps side by side."""
        vmax = np.nanpercentile(original_depth, percentile)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original_depth, cmap='inferno_r', vmin=0, vmax=vmax)
        plt.title('Original Depth Map')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(filled_depth, cmap='inferno_r', vmin=0, vmax=vmax)
        plt.title('Filled Depth Map')
        plt.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


class DepthMapDensifier:
    """Main class that orchestrates the entire depth map densification pipeline."""
    
    def __init__(self):
        self.colormap_handler = SemanticColormap()
        self.depth_processor = DepthProcessor(self.colormap_handler)
        self.depth_filler = DepthFiller(self.colormap_handler)
        self.depth_smoother = DepthSmoother(self.colormap_handler)
        self.depth_analyzer = DepthAnalyzer(self.colormap_handler)
    
    def process_depth_with_segmentation(self, instance_image_path: str, original_image: np.ndarray,
                                      depth_image_path: str, depth_npy_path: str,
                                      segmentation_image_path: str, method: str = 'semantic_joint') -> Tuple:
        """
        Main function to process depth image using segmentation.
        
        Args:
            instance_image_path: Path to instance segmentation image
            original_image: Original RGB image
            depth_image_path: Path to depth image with inferno colormap
            depth_npy_path: Path to depth numpy array
            segmentation_image_path: Path to segmentation image
            method: Filling method ('semantic_joint', 'knn')
            
        Returns:
            Tuple of (original_depth, filled_depth, segmentation_map, stats)
        """
        # Load images
        depth_color_image = cv2.imread(depth_image_path)
        depth_color_image = cv2.cvtColor(depth_color_image, cv2.COLOR_BGR2RGB)
        
        instance_image = cv2.imread(instance_image_path)
        instance_image = cv2.cvtColor(instance_image, cv2.COLOR_BGR2RGB)
        
        depth_npy = np.load(depth_npy_path, allow_pickle=True)
        
        # Ensure consistent dimensions
        depth_height, depth_width = depth_color_image.shape[:2]
        if instance_image.shape[0] != depth_height or instance_image.shape[1] != depth_width:
            print(f"Resizing instance image from {instance_image.shape[:2]} to {depth_height}x{depth_width}")
            instance_image = self.depth_processor.resize_segmentation(instance_image, depth_height, depth_width)
        
        # Decode segmentation image to class IDs
        segmentation_map = self.depth_processor.decode_segmentation_map(instance_image)
        
        # Use numpy depth data
        depth_map = depth_npy
        
        # Fill missing values based on method
        if method == 'semantic_joint':
            filled_depth = self.depth_filler.semantic_joint_fill_depth(
                depth_map, segmentation_map, self.colormap_handler.dynamic_class_ids
            )
        elif method == 'knn':
            filled_depth = self.depth_filler.knn_fill_missing_depths(depth_map, segmentation_map)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Analyze depth by class
        stats = self.depth_analyzer.analyze_depth_by_class(depth_map, segmentation_map)
        
        # Visualize results
        filled_depth_vis = 1 / filled_depth
        self.depth_analyzer.visualize_results(
            original_image, depth_color_image, filled_depth_vis, 
            segmentation_map, self.colormap_handler.reduced_colormap
        )
        
        # Calculate completion statistics
        total_pixels = depth_map.size
        missing_before = np.sum(np.isnan(depth_map))
        missing_after = np.sum(np.isnan(filled_depth))
        completion_rate = ((missing_before - missing_after) / missing_before) * 100 if missing_before > 0 else 100
        
        print(f"Depth map completion statistics:")
        print(f"  Total pixels: {total_pixels}")
        print(f"  Missing pixels before: {missing_before} ({missing_before/total_pixels*100:.2f}%)")
        print(f"  Missing pixels after: {missing_after} ({missing_after/total_pixels*100:.2f}%)")
        print(f"  Completion rate: {completion_rate:.2f}%")
        
        return depth_map, filled_depth, segmentation_map, stats
    
    def smooth_filled_depth(self, filled_depth: np.ndarray, segmentation_map: np.ndarray,
                           method: str = 'guided_segmentation', **kwargs) -> np.ndarray:
        """
        Apply smoothing to filled depth map.
        
        Args:
            filled_depth: Filled depth map
            segmentation_map: Segmentation map
            method: Smoothing method ('guided_segmentation' or 'connected_objects')
            **kwargs: Additional parameters for smoothing methods
            
        Returns:
            Smoothed depth map
        """
        if method == 'guided_segmentation':
            return self.depth_smoother.smooth_depth_with_guided_segmentation(
                filled_depth, segmentation_map, **kwargs
            )
        elif method == 'connected_objects':
            return self.depth_smoother.smooth_depth_connected_objects(
                filled_depth, segmentation_map, **kwargs
            )
        else:
            raise ValueError(f"Unknown smoothing method: {method}")


def main():
    """
    Main function demonstrating usage of the depth map densification pipeline.
    """
    # Initialize the densification system
    densifier = DepthMapDensifier()
    
    # Configuration
    image_number = 68  # Change this to process different images
    
    # File paths
    instance_image_path = f"./seg_map/seg_map_{image_number}.png"
    depth_image_path = f"./res/depth_map_{image_number}.png"
    depth_npy_path = f"./res_npy/depth_map_{image_number}.npy"
    segmentation_image_path = f"./seg_map/seg_map_{image_number}.png"
    original_image_path = f'./images1/{image_number}.png'
    
    # Check if files exist
    for path in [instance_image_path, depth_image_path, depth_npy_path, original_image_path]:
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            return
    
    # Load original image
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    print("Starting depth map densification...")
    
    # Process with semantic joint method
    depth_map, filled_depth, segmentation_map, stats = densifier.process_depth_with_segmentation(
        instance_image_path, original_image, depth_image_path, 
        depth_npy_path, segmentation_image_path, method='semantic_joint'
    )
    
    # Apply smoothing
    print("Applying smoothing...")
    smoothed_depth = densifier.smooth_filled_depth(
        filled_depth, segmentation_map, method='guided_segmentation',
        sigma_spatial=1, sigma_depth=0.5
    )
    
    # Visualize final results
    densifier.depth_analyzer.plot_depth_comparison(filled_depth, smoothed_depth, "Filled vs Smoothed Depth")
    densifier.depth_analyzer.plot_depth_histogram(smoothed_depth, "Smoothed Depth Histogram")
    
    # Save results
    output_dir = "output_results"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.imsave(f"{output_dir}/filled_depth_{image_number}.png", filled_depth, cmap='inferno')
    plt.imsave(f"{output_dir}/smoothed_depth_{image_number}.png", smoothed_depth, cmap='inferno')
    
    print(f"Results saved to {output_dir}/")
    print("Depth map densification completed successfully!")


if __name__ == "__main__":
    main()
