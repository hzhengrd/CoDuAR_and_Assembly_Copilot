# Single-Stream Compositional Dual-hand VideoMAE with Transformer Decoder
# Uses SAME video input for both left and right hand predictions
# Shared encoder + separate hand-specific adapters and decoders + separate classification heads

import torch
import torch.nn as nn
from timm.models.registry import register_model
from .modeling_finetune import VisionTransformer
from .modeling_finetune_compositional_dual_transformer import (
    HandSpecificAdapter,
    ElementTransformerDecoder
)


class CompositionalDualHeadTransformerSingleStream(VisionTransformer):
    """
    Single-Stream Vision Transformer with Transformer Decoder for dual-hand classification.
    
    Key Features:
        - SINGLE video input (shared encoder)
        - Separate hand-specific adapters
        - Separate transformer decoders for each hand
        - Element queries communicate via self-attention
        - 8 total classification heads (4 per hand)
    
    Architecture:
        Single Video Input
              ↓
        Shared Encoder
           /     \
      LH Adapter  RH Adapter
          ↓         ↓
      LH Decoder  RH Decoder
        (4 elem)  (4 elem)
          ↓         ↓
      LH Heads    RH Heads
    """
    
    def __init__(
        self,
        lh_num_verbs=20,
        lh_num_manip_objs=50,
        lh_num_target_objs=50,
        lh_num_tools=20,
        rh_num_verbs=20,
        rh_num_manip_objs=50,
        rh_num_target_objs=50,
        rh_num_tools=20,
        use_hand_adapters=True,
        adapter_dim=128,
        decoder_layers=3,
        decoder_heads=8,
        decoder_dim=2048,
        decoder_dropout=0.1,
        head_dropout=0.1,
        **kwargs
    ):
        # Remove num_classes from kwargs if present
        kwargs.pop('num_classes', None)
        super().__init__(num_classes=1000, **kwargs)
        
        # Remove the single head
        embed_dim = self.embed_dim
        del self.head
        
        # Hand-specific adapters
        self.use_hand_adapters = use_hand_adapters
        if use_hand_adapters:
            self.lh_adapter = HandSpecificAdapter(embed_dim, adapter_dim)
            self.rh_adapter = HandSpecificAdapter(embed_dim, adapter_dim)
        
        # Transformer decoders for element communication (one per hand)
        self.lh_decoder = ElementTransformerDecoder(
            num_elements=4,
            d_model=embed_dim,
            nhead=decoder_heads,
            num_layers=decoder_layers,
            dim_feedforward=decoder_dim,
            dropout=decoder_dropout
        )
        
        self.rh_decoder = ElementTransformerDecoder(
            num_elements=4,
            d_model=embed_dim,
            nhead=decoder_heads,
            num_layers=decoder_layers,
            dim_feedforward=decoder_dim,
            dropout=decoder_dropout
        )
        
        # Left-hand classification heads (4 elements)
        self.lh_verb_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, lh_num_verbs)
        )
        self.lh_manip_obj_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, lh_num_manip_objs)
        )
        self.lh_target_obj_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, lh_num_target_objs)
        )
        self.lh_tool_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, lh_num_tools)
        )
        
        # Right-hand classification heads (4 elements)
        self.rh_verb_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, rh_num_verbs)
        )
        self.rh_manip_obj_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, rh_num_manip_objs)
        )
        self.rh_target_obj_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, rh_num_target_objs)
        )
        self.rh_tool_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, rh_num_tools)
        )
    
    def forward(self, x):
        """
        Args:
            x: dict with 'frames' key (single video input)
               OR tensor (for backward compatibility)
        
        Returns:
            dict with predictions for all 8 elements (4 per hand)
        """
        # Handle both dict and tensor input
        if isinstance(x, dict):
            frames = x['frames']
        else:
            frames = x
        
        # Process through shared encoder ONCE
        shared_features = self.forward_features(frames)  # [batch, 768]
        
        # === LEFT HAND BRANCH ===
        lh_features = shared_features
        if self.use_hand_adapters:
            lh_features = self.lh_adapter(lh_features)
        
        # Transformer decoder: elements communicate
        lh_element_features = self.lh_decoder(lh_features)  # [batch, 4, 768]
        
        # Extract element-specific features
        lh_verb_feat = lh_element_features[:, 0, :]
        lh_manip_feat = lh_element_features[:, 1, :]
        lh_target_feat = lh_element_features[:, 2, :]
        lh_tool_feat = lh_element_features[:, 3, :]
        
        # Predict each element
        lh_verb = self.lh_verb_head(lh_verb_feat)
        lh_manip_obj = self.lh_manip_obj_head(lh_manip_feat)
        lh_target_obj = self.lh_target_obj_head(lh_target_feat)
        lh_tool = self.lh_tool_head(lh_tool_feat)
        
        # === RIGHT HAND BRANCH ===
        rh_features = shared_features
        if self.use_hand_adapters:
            rh_features = self.rh_adapter(rh_features)
        
        # Transformer decoder: elements communicate
        rh_element_features = self.rh_decoder(rh_features)  # [batch, 4, 768]
        
        # Extract element-specific features
        rh_verb_feat = rh_element_features[:, 0, :]
        rh_manip_feat = rh_element_features[:, 1, :]
        rh_target_feat = rh_element_features[:, 2, :]
        rh_tool_feat = rh_element_features[:, 3, :]
        
        # Predict each element
        rh_verb = self.rh_verb_head(rh_verb_feat)
        rh_manip_obj = self.rh_manip_obj_head(rh_manip_feat)
        rh_target_obj = self.rh_target_obj_head(rh_target_feat)
        rh_tool = self.rh_tool_head(rh_tool_feat)
        
        return {
            # Left hand predictions
            'lh_verb': lh_verb,
            'lh_manip_obj': lh_manip_obj,
            'lh_target_obj': lh_target_obj,
            'lh_tool': lh_tool,
            # Right hand predictions
            'rh_verb': rh_verb,
            'rh_manip_obj': rh_manip_obj,
            'rh_target_obj': rh_target_obj,
            'rh_tool': rh_tool,
        }


@register_model
def vit_base_patch16_224_compositional_dual_transformer_single_stream(pretrained=False, **kwargs):
    """
    ViT-Base model for single-stream compositional dual-hand action recognition.
    
    Single video input → Shared encoder → Separate hand branches → 8 classification heads
    """
    model = CompositionalDualHeadTransformerSingleStream(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=nn.LayerNorm, **kwargs)
    return model


@register_model
def vit_large_patch16_224_compositional_dual_transformer_single_stream(pretrained=False, **kwargs):
    """
    ViT-Large model for single-stream compositional dual-hand action recognition.
    """
    model = CompositionalDualHeadTransformerSingleStream(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=nn.LayerNorm, **kwargs)
    return model

