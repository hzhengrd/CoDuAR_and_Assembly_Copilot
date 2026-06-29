# Compositional Dual-hand VideoMAE with Transformer Decoder
# Elements communicate via self-attention for better compositional reasoning
import torch
import torch.nn as nn
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from .modeling_finetune import VisionTransformer


class HandSpecificAdapter(nn.Module):
    """Hand-specific adapter module to capture hand-specific features."""
    def __init__(self, embed_dim, bottleneck_dim=128):
        super().__init__()
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        
        # Initialize to near-identity
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.01)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.01)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return residual + x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer for element communication.

    Flow (full):
        1. Self-attention between element queries  [optional, controlled by use_self_attn]
        2. Cross-attention to encoder features
        3. Feed-forward network

    Setting use_self_attn=False removes the self-attention sub-block entirely
    (parameters are not allocated), leaving only cross-attention + FFN.
    This is used for the ablation study on element-wise self-attention.
    """
    def __init__(self, d_model=768, nhead=8, dim_feedforward=2048, dropout=0.1,
                 use_self_attn=True):
        super().__init__()

        self.use_self_attn = use_self_attn

        # Self-attention (elements talk to each other) — omitted in ablation
        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Cross-attention (elements attend to encoder features)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, queries, encoder_features):
        """
        Args:
            queries: [batch, num_elements, d_model] - Element queries
            encoder_features: [batch, 1, d_model] - Encoder CLS token

        Returns:
            queries: [batch, num_elements, d_model] - Updated element representations
        """
        # Transpose for MultiheadAttention: [batch, seq, d_model] -> [seq, batch, d_model]
        queries_t = queries.transpose(0, 1)
        encoder_features_t = encoder_features.transpose(0, 1)

        # Self-attention: elements communicate with each other (skipped in ablation)
        if self.use_self_attn:
            attn_output, _ = self.self_attn(queries_t, queries_t, queries_t)
            queries_t = queries_t + self.dropout1(attn_output)
            queries_t = self.norm1(queries_t)

        # Cross-attention: elements attend to encoder features
        attn_output, _ = self.cross_attn(queries_t, encoder_features_t, encoder_features_t)
        queries_t = queries_t + self.dropout2(attn_output)
        queries_t = self.norm2(queries_t)

        # Feed-forward
        ffn_output = self.ffn(queries_t)
        queries_t = queries_t + ffn_output
        queries_t = self.norm3(queries_t)

        # Transpose back to [batch, seq, d_model]
        queries = queries_t.transpose(0, 1)

        return queries


class ElementTransformerDecoder(nn.Module):
    """
    Multi-layer transformer decoder for compositional element prediction.
    
    Each element has a learnable query that:
    1. Attends to other elements (learn dependencies)
    2. Attends to encoder features (extract relevant visual info)
    3. Produces element-specific representation
    """
    def __init__(self, num_elements=4, d_model=768, nhead=8, num_layers=3,
                 dim_feedforward=2048, dropout=0.1, use_self_attn=True):
        super().__init__()

        # Learnable element queries (verb, manip_obj, target_obj, tool)
        self.element_queries = nn.Parameter(torch.randn(1, num_elements, d_model))
        nn.init.xavier_uniform_(self.element_queries)

        # Stack of decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                    use_self_attn=use_self_attn)
            for _ in range(num_layers)
        ])
        
        self.num_elements = num_elements
        self.d_model = d_model
    
    def forward(self, encoder_features):
        """
        Args:
            encoder_features: [batch, d_model] - CLS token from encoder
        
        Returns:
            element_features: [batch, num_elements, d_model]
        """
        batch_size = encoder_features.size(0)
        
        # Expand element queries for batch
        queries = self.element_queries.expand(batch_size, -1, -1)  # [batch, 4, 768]
        
        # Expand encoder features for cross-attention
        encoder_features = encoder_features.unsqueeze(1)  # [batch, 1, 768]
        
        # Pass through decoder layers
        for layer in self.layers:
            queries = layer(queries, encoder_features)
        
        return queries  # [batch, 4, 768]


class CompositionalDualHeadTransformer(VisionTransformer):
    """
    Vision Transformer with Transformer Decoder for compositional dual-hand classification.
    
    Key Innovation:
        - Element queries communicate via self-attention
        - Learn element dependencies (e.g., verb influences object prediction)
        - More sophisticated compositional reasoning
    
    Architecture:
        Encoder → Hand Adapter → Transformer Decoder → Element Heads
                                     ↑
                                Self-attention between elements!
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
        use_shared_adapter=False,
        adapter_dim=128,
        decoder_layers=3,
        decoder_heads=8,
        decoder_dim=2048,
        decoder_dropout=0.1,
        use_element_self_attn=True,
        head_dropout=0.1,
        **kwargs
    ):
        # Remove num_classes from kwargs if present
        kwargs.pop('num_classes', None)
        super().__init__(num_classes=1000, **kwargs)
        
        # Remove the single head
        embed_dim = self.embed_dim
        del self.head

        # Adapter configuration (mutually exclusive modes):
        #   use_hand_adapters=True, use_shared_adapter=False  -> two separate hand-specific adapters
        #   use_hand_adapters=False, use_shared_adapter=True  -> one shared adapter for both hands
        #   use_hand_adapters=False, use_shared_adapter=False -> no adapter (ablation: adapter removed)
        self.use_hand_adapters = use_hand_adapters
        self.use_shared_adapter = use_shared_adapter
        if use_hand_adapters:
            self.lh_adapter = HandSpecificAdapter(embed_dim, adapter_dim)
            self.rh_adapter = HandSpecificAdapter(embed_dim, adapter_dim)
        elif use_shared_adapter:
            self.shared_adapter = HandSpecificAdapter(embed_dim, adapter_dim)
        
        # Transformer decoders for element communication
        self.lh_decoder = ElementTransformerDecoder(
            num_elements=4,
            d_model=embed_dim,
            nhead=decoder_heads,
            num_layers=decoder_layers,
            dim_feedforward=decoder_dim,
            dropout=decoder_dropout,
            use_self_attn=use_element_self_attn,
        )

        self.rh_decoder = ElementTransformerDecoder(
            num_elements=4,
            d_model=embed_dim,
            nhead=decoder_heads,
            num_layers=decoder_layers,
            dim_feedforward=decoder_dim,
            dropout=decoder_dropout,
            use_self_attn=use_element_self_attn,
        )
        
        # Classification heads (one per element)
        # Left hand
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
        
        # Right hand
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
        
        # Initialize heads
        for head in [self.lh_verb_head, self.lh_manip_obj_head, self.lh_target_obj_head, self.lh_tool_head,
                    self.rh_verb_head, self.rh_manip_obj_head, self.rh_target_obj_head, self.rh_tool_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        # Store configuration
        self.lh_num_verbs = lh_num_verbs
        self.lh_num_manip_objs = lh_num_manip_objs
        self.lh_num_target_objs = lh_num_target_objs
        self.lh_num_tools = lh_num_tools
        self.rh_num_verbs = rh_num_verbs
        self.rh_num_manip_objs = rh_num_manip_objs
        self.rh_num_target_objs = rh_num_target_objs
        self.rh_num_tools = rh_num_tools
        self.compositional_mode = True
    
    def forward(self, x):
        """
        Args:
            x: dict with 'lh_frames' and 'rh_frames'
        
        Returns:
            dict with predictions for all 8 elements
        """
        if not isinstance(x, dict):
            raise ValueError("CompositionalDualHeadTransformer expects dict input")
        
        lh_x = x['lh_frames']
        rh_x = x['rh_frames']
        
        # Process left hand through encoder
        lh_features = self.forward_features(lh_x)  # [batch, 768]
        
        # Apply adapter
        if self.use_hand_adapters:
            lh_features = self.lh_adapter(lh_features)
        elif self.use_shared_adapter:
            lh_features = self.shared_adapter(lh_features)
        
        # Transformer decoder: elements communicate
        lh_element_features = self.lh_decoder(lh_features)  # [batch, 4, 768]
        
        # Extract element-specific features
        # Element order: [verb, manip_obj, target_obj, tool]
        lh_verb_feat = lh_element_features[:, 0, :]      # [batch, 768]
        lh_manip_feat = lh_element_features[:, 1, :]     # [batch, 768]
        lh_target_feat = lh_element_features[:, 2, :]    # [batch, 768]
        lh_tool_feat = lh_element_features[:, 3, :]      # [batch, 768]
        
        # Left hand predictions
        lh_verb_pred = self.lh_verb_head(lh_verb_feat)
        lh_manip_obj_pred = self.lh_manip_obj_head(lh_manip_feat)
        lh_target_obj_pred = self.lh_target_obj_head(lh_target_feat)
        lh_tool_pred = self.lh_tool_head(lh_tool_feat)
        
        # Process right hand through encoder
        rh_features = self.forward_features(rh_x)
        
        # Apply adapter
        if self.use_hand_adapters:
            rh_features = self.rh_adapter(rh_features)
        elif self.use_shared_adapter:
            rh_features = self.shared_adapter(rh_features)
        
        # Transformer decoder: elements communicate
        rh_element_features = self.rh_decoder(rh_features)  # [batch, 4, 768]
        
        # Extract element-specific features
        rh_verb_feat = rh_element_features[:, 0, :]
        rh_manip_feat = rh_element_features[:, 1, :]
        rh_target_feat = rh_element_features[:, 2, :]
        rh_tool_feat = rh_element_features[:, 3, :]
        
        # Right hand predictions
        rh_verb_pred = self.rh_verb_head(rh_verb_feat)
        rh_manip_obj_pred = self.rh_manip_obj_head(rh_manip_feat)
        rh_target_obj_pred = self.rh_target_obj_head(rh_target_feat)
        rh_tool_pred = self.rh_tool_head(rh_tool_feat)
        
        outputs = {
            'lh_verb': lh_verb_pred,
            'lh_manip_obj': lh_manip_obj_pred,
            'lh_target_obj': lh_target_obj_pred,
            'lh_tool': lh_tool_pred,
            'rh_verb': rh_verb_pred,
            'rh_manip_obj': rh_manip_obj_pred,
            'rh_target_obj': rh_target_obj_pred,
            'rh_tool': rh_tool_pred,
        }
        
        return outputs
    
    def extract_element_features(self, x):
        """
        Extract element-specific features for analysis/visualization.
        
        Returns:
            dict with element features for each hand
        """
        if not isinstance(x, dict):
            raise ValueError("Expected dict input")
        
        lh_x = x['lh_frames']
        rh_x = x['rh_frames']
        
        # Left hand
        lh_features = self.forward_features(lh_x)
        if self.use_hand_adapters:
            lh_features = self.lh_adapter(lh_features)
        elif self.use_shared_adapter:
            lh_features = self.shared_adapter(lh_features)
        lh_element_features = self.lh_decoder(lh_features)
        
        # Right hand
        rh_features = self.forward_features(rh_x)
        if self.use_hand_adapters:
            rh_features = self.rh_adapter(rh_features)
        elif self.use_shared_adapter:
            rh_features = self.shared_adapter(rh_features)
        rh_element_features = self.rh_decoder(rh_features)
        
        return {
            'lh_verb_features': lh_element_features[:, 0, :],
            'lh_manip_features': lh_element_features[:, 1, :],
            'lh_target_features': lh_element_features[:, 2, :],
            'lh_tool_features': lh_element_features[:, 3, :],
            'rh_verb_features': rh_element_features[:, 0, :],
            'rh_manip_features': rh_element_features[:, 1, :],
            'rh_target_features': rh_element_features[:, 2, :],
            'rh_tool_features': rh_element_features[:, 3, :],
        }
    
    def get_classifier(self):
        return {
            'lh_verb_head': self.lh_verb_head,
            'lh_manip_obj_head': self.lh_manip_obj_head,
            'lh_target_obj_head': self.lh_target_obj_head,
            'lh_tool_head': self.lh_tool_head,
            'rh_verb_head': self.rh_verb_head,
            'rh_manip_obj_head': self.rh_manip_obj_head,
            'rh_target_obj_head': self.rh_target_obj_head,
            'rh_tool_head': self.rh_tool_head,
        }


@register_model
def vit_base_patch16_224_compositional_dual_transformer(pretrained=False, **kwargs):
    """ViT-Base model with Transformer decoder for element communication"""
    model = CompositionalDualHeadTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=nn.LayerNorm, **kwargs)
    model.default_cfg = {
        'url': '',
        'num_classes': 'compositional_transformer',
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .9,
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
    }
    return model


@register_model
def vit_base_patch16_224_compositional_dual_transformer_no_self_attn(pretrained=False, **kwargs):
    """ViT-Base with hand-specific adapters but NO element self-attention in decoders.

    Ablation study: isolates the contribution of element-wise self-attention.
    Cross-attention (elements → encoder) and FFN are kept intact.
    """
    kwargs['use_element_self_attn'] = False
    model = CompositionalDualHeadTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=nn.LayerNorm, **kwargs)
    model.default_cfg = {
        'url': '',
        'num_classes': 'compositional_transformer_no_self_attn',
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .9,
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
    }
    return model


@register_model
def vit_base_patch16_224_compositional_dual_transformer_shared_adapter(pretrained=False, **kwargs):
    """ViT-Base with a single shared adapter for both hands (ablation: no hand-specific adapters)."""
    kwargs['use_hand_adapters'] = False
    kwargs['use_shared_adapter'] = True
    model = CompositionalDualHeadTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=nn.LayerNorm, **kwargs)
    model.default_cfg = {
        'url': '',
        'num_classes': 'compositional_transformer_shared_adapter',
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .9,
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
    }
    return model


@register_model
def vit_large_patch16_224_compositional_dual_transformer(pretrained=False, **kwargs):
    """ViT-Large model with Transformer decoder for element communication"""
    model = CompositionalDualHeadTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=nn.LayerNorm, **kwargs)
    model.default_cfg = {
        'url': '',
        'num_classes': 'compositional_transformer',
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .9,
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
    }
    return model

