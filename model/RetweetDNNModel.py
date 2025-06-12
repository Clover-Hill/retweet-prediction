import pdb 
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import PreTrainedModel, PretrainedConfig
from transformers import Qwen3Config, Qwen3Model, Qwen3PreTrainedModel
from transformers.activations import ACT2FN

class RetweetFeatureConfig(PretrainedConfig):
    """Configuration class for RetweetFeatureModel"""
    
    model_type = "retweet_feature"
    
    def __init__(
        self,
        dense_features_dim: int = 100,
        sparse_feature_dims: Optional[Dict[str, int]] = None,
        varlen_feature_dims: Optional[Dict[str, int]] = None,
        sparse_feature_names: Optional[List[str]] = None,
        varlen_feature_names: Optional[List[str]] = None,
        embedding_dim: int = 32,
        dnn_hidden_units: List[int] = None,
        dnn_dropout: float = 0.3,
        dnn_activation: str = 'relu',
        use_bn: bool = True,
        l2_reg: float = 1e-4,
        init_std: float = 1e-4,
        varlen_pooling_modes: List[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Set defaults
        if sparse_feature_dims is None:
            sparse_feature_dims = {}
        if varlen_feature_dims is None:
            varlen_feature_dims = {}
        if sparse_feature_names is None:
            sparse_feature_names = []
        if varlen_feature_names is None:
            varlen_feature_names = []
        if dnn_hidden_units is None:
            dnn_hidden_units = [2048, 512, 128]
        if varlen_pooling_modes is None:
            varlen_pooling_modes = ['mean', 'max']
        
        self.dense_features_dim = dense_features_dim
        self.sparse_feature_dims = sparse_feature_dims
        self.varlen_feature_dims = varlen_feature_dims
        self.sparse_feature_names = sparse_feature_names
        self.varlen_feature_names = varlen_feature_names
        self.embedding_dim = embedding_dim
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_dropout = dnn_dropout
        self.dnn_activation = dnn_activation
        self.use_bn = use_bn
        self.l2_reg = l2_reg
        self.init_std = init_std
        self.varlen_pooling_modes = varlen_pooling_modes

class DNN(nn.Module):
    """Deep Neural Network module with batch normalization and dropout"""
    
    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, 
                 use_bn=False, init_std=0.0001, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!")
        
        hidden_units = [inputs_dim] + list(hidden_units)
        self.linears = nn.ModuleList([
            nn.Linear(hidden_units[i], hidden_units[i + 1]) 
            for i in range(len(hidden_units) - 1)
        ])
        
        if self.use_bn:
            self.bn = nn.ModuleList([
                nn.BatchNorm1d(hidden_units[i + 1]) 
                for i in range(len(hidden_units) - 1)
            ])
        
        self.activation = activation
        
        # Initialize weights
        for linear in self.linears:
            nn.init.normal_(linear.weight, mean=0, std=init_std)
    
    def forward(self, inputs):
        deep_input = inputs
        
        for i, linear in enumerate(self.linears):
            fc = linear(deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = F.relu(fc) if self.activation == 'relu' else fc
            fc = self.dropout(fc)
            deep_input = fc
        
        return deep_input


class FeatureEmbedding(nn.Module):
    """Embedding module for sparse features"""
    
    def __init__(self, feature_dims: Dict[str, int], embedding_dim: int = 32):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(num_embeddings=dim, embedding_dim=embedding_dim)
            for feat, dim in feature_dims.items()
        })
    
    def forward(self, sparse_features: torch.Tensor, feature_names: List[str]) -> torch.Tensor:
        """
        Args:
            sparse_features: [batch_size, num_features]
            feature_names: List of feature names corresponding to columns
        
        Returns:
            Embedded features: [batch_size, num_features * embedding_dim]
        """
        embedded_features = []
        for i, feat_name in enumerate(feature_names):
            if feat_name in self.embeddings:
                embedded = self.embeddings[feat_name](sparse_features[:, i])
                embedded_features.append(embedded)
        
        return torch.cat(embedded_features, dim=-1)


class VarlenPooling(nn.Module):
    """Pooling module for variable-length features"""
    
    def __init__(self, mode='mean'):
        super().__init__()
        self.mode = mode
    
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            embeddings: [batch_size, seq_len, embedding_dim]
            mask: [batch_size, seq_len] - 0 for padding, 1 for valid
        
        Returns:
            Pooled embeddings: [batch_size, embedding_dim]
        """
        if mask is None:
            mask = (embeddings.sum(dim=-1) != 0).float()
        
        if self.mode == 'mean':
            mask_expanded = mask.unsqueeze(-1)
            sum_embeddings = (embeddings * mask_expanded).sum(dim=1)
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            return sum_embeddings / lengths
        elif self.mode == 'sum':
            mask_expanded = mask.unsqueeze(-1)
            return (embeddings * mask_expanded).sum(dim=1)
        elif self.mode == 'max':
            # Replace padding with -inf for max pooling
            mask_expanded = mask.unsqueeze(-1)
            # Use a large negative value instead of -inf to prevent NaN
            embeddings_masked = embeddings.masked_fill(~mask_expanded.bool(), -1e9)
            pooled = embeddings_masked.max(dim=1)[0]
            # Replace -1e9 with 0 for all-padding sequences
            all_padding = (~mask.bool()).all(dim=1, keepdim=True)
            pooled = pooled.masked_fill(all_padding, 0)
            return pooled
        else:
            raise ValueError(f"Unsupported pooling mode: {self.mode}")


class RetweetFeatureModel(PreTrainedModel):
    """Feature-only model for retweet prediction with Transformers integration"""
    
    config_class = RetweetFeatureConfig
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def __init__(self, config: RetweetFeatureConfig):
        super().__init__(config)
        
        self.config = config
        
        # Sparse feature embeddings
        self.sparse_embedding = FeatureEmbedding(config.sparse_feature_dims, config.embedding_dim)
        
        # Variable-length feature embeddings
        self.varlen_embeddings = nn.ModuleDict({
            feat: nn.Embedding(num_embeddings=dim, embedding_dim=config.embedding_dim)
            for feat, dim in config.varlen_feature_dims.items()
        })
        
        # Pooling layers for variable-length features
        self.varlen_pooling = nn.ModuleDict({
            mode: VarlenPooling(mode) for mode in config.varlen_pooling_modes
        })
        
        # Calculate DNN input dimension
        dnn_input_dim = (
            config.dense_features_dim +  # Dense features
            len(config.sparse_feature_names) * config.embedding_dim +  # Sparse embeddings
            len(config.varlen_feature_names) * len(config.varlen_pooling_modes) * config.embedding_dim  # Varlen embeddings
        )
        
        # DNN layers
        self.dnn = DNN(
            inputs_dim=dnn_input_dim,
            hidden_units=config.dnn_hidden_units,
            activation=config.dnn_activation,
            l2_reg=config.l2_reg,
            dropout_rate=config.dnn_dropout,
            use_bn=config.use_bn,
            init_std=config.init_std,
            device=self.device
        )
        
        # Output layer
        self.output_layer = nn.Linear(config.dnn_hidden_units[-1], 1)
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: torch.Tensor,
        varlen_features: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutput]:
        """
        Forward pass
        
        Args:
            dense_features: [batch_size, num_dense_features]
            sparse_features: [batch_size, num_sparse_features]
            varlen_features: Dict of feature_name -> [batch_size, seq_len]
            labels: [batch_size]
            return_dict: Whether to return a SequenceClassifierOutput instead of tuple
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Embed sparse features
        sparse_embedded = self.sparse_embedding(sparse_features, self.config.sparse_feature_names)
        
        # Process variable-length features
        varlen_pooled_list = []
        for feat_name in self.config.varlen_feature_names:
            if feat_name in varlen_features:
                varlen_tensor = varlen_features[feat_name]
                # Get embeddings
                embedded = self.varlen_embeddings[feat_name](varlen_tensor)
                # Create mask (0 is padding)
                mask = (varlen_tensor != 0).float()
                # Apply different pooling modes
                for mode in self.config.varlen_pooling_modes:
                    pooled = self.varlen_pooling[mode](embedded, mask)
                    varlen_pooled_list.append(pooled)
        
        # Concatenate all features
        if varlen_pooled_list:
            varlen_pooled = torch.cat(varlen_pooled_list, dim=-1)
            dnn_input = torch.cat([dense_features, sparse_embedded, varlen_pooled], dim=-1)
        else:
            dnn_input = torch.cat([dense_features, sparse_embedded], dim=-1)
        
        # Pass through DNN
        dnn_output = self.dnn(dnn_input)
        
        # Get predictions
        logits = self.output_layer(dnn_output).squeeze(-1)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # MSE loss for regression
            loss = F.l1_loss(logits, labels)
        
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )

# Qwen3 related classes (from your previous code)
class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Extract embeddings from the last token position."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class RetweetFusionConfig(Qwen3Config):
    """Configuration for fusion model"""
    
    def __init__(
        self,
        dense_features_dim: int = 89,  # Based on your feature list
        sparse_features_dim: int = 26,  # Based on your feature list
        embedding_dim: int = 32,
        mlp_num: int = 4,
        fusion_hidden_size: int = 1024,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dense_features_dim = dense_features_dim
        self.sparse_features_dim = sparse_features_dim
        self.embedding_dim = embedding_dim
        self.mlp_num = mlp_num
        self.fusion_hidden_size = fusion_hidden_size
        self.dropout_rate = dropout_rate

class RetweetFusionModel(Qwen3PreTrainedModel):
    """Fusion model combining Qwen3 text embeddings with engineered features"""
    
    config_class = RetweetFusionConfig
    base_model_prefix = "qwen_model"
    
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3RMSNorm):
            module.weight.data.fill_(1.0)

    def __init__(self, config: RetweetFusionConfig):
        super().__init__(config)
        self.config = config
        
        # Qwen3 model for text encoding
        self.qwen_model = Qwen3Model(config)
        
        # Feature model components (simplified - you'd use the full feature dims from your data)
        sparse_feature_dims = {f'feat_{i}': 100 for i in range(config.sparse_features_dim)}
        varlen_feature_dims = {'user_mentions': 10000, 'urls': 10000, 'hashtags': 10000}
        
        # Calculate feature embedding dimension
        feature_embedding_dim = (
            config.dense_features_dim +
            config.sparse_features_dim * config.embedding_dim +
            3 * 2 * config.embedding_dim  # 3 varlen features, 2 pooling modes
        )
        
        # Projection layers to align dimensions
        self.text_projection = nn.Linear(config.hidden_size, config.fusion_hidden_size)
        self.feature_projection = nn.Linear(feature_embedding_dim, config.fusion_hidden_size)
        
        # Feature embeddings
        self.sparse_embedding = FeatureEmbedding(sparse_feature_dims, config.embedding_dim)
        self.varlen_embeddings = nn.ModuleDict({
            feat: nn.Embedding(num_embeddings=dim, embedding_dim=config.embedding_dim)
            for feat, dim in varlen_feature_dims.items()
        })
        self.varlen_pooling = nn.ModuleDict({
            mode: VarlenPooling(mode) for mode in ['mean', 'max']
        })
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        fusion_mlp_config = type('MLPConfig', (), {
            'hidden_size': config.fusion_hidden_size,
            'intermediate_size': config.fusion_hidden_size * 4,
            'hidden_act': config.hidden_act
        })()
        
        for _ in range(config.mlp_num):
            self.fusion_layers.append(Qwen3MLP(fusion_mlp_config))
            self.layer_norms.append(Qwen3RMSNorm(config.fusion_hidden_size, eps=config.rms_norm_eps))
        
        self.final_norm = Qwen3RMSNorm(config.fusion_hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Output layer
        self.regression_head = nn.Linear(config.fusion_hidden_size, 1)
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        dense_features: Optional[torch.Tensor] = None,
        sparse_features: Optional[torch.Tensor] = None,
        varlen_features: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> SequenceClassifierOutput:
        """
        Forward pass combining text and features
        """
        # Process text through Qwen3
        if input_ids is not None and attention_mask is not None:
            qwen_outputs = self.qwen_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            text_embeddings = last_token_pool(qwen_outputs.last_hidden_state, attention_mask)
            text_projected = self.text_projection(text_embeddings)
        else:
            # If no text, use zeros
            batch_size = dense_features.size(0)
            text_projected = torch.zeros(batch_size, self.config.fusion_hidden_size, device=self.device)
        
        # Process features
        # Create dummy sparse feature names (in practice, use actual names)
        sparse_feature_names = [f'feat_{i}' for i in range(sparse_features.size(1))]
        sparse_embedded = self.sparse_embedding(sparse_features, sparse_feature_names)
        
        # Process variable-length features
        varlen_pooled_list = []
        for feat_name in ['user_mentions', 'urls', 'hashtags']:
            if feat_name in varlen_features:
                varlen_tensor = varlen_features[feat_name]
                embedded = self.varlen_embeddings[feat_name](varlen_tensor)
                mask = (varlen_tensor != 0).float()
                for mode in ['mean', 'max']:
                    pooled = self.varlen_pooling[mode](embedded, mask)
                    varlen_pooled_list.append(pooled)
        
        # Concatenate all features
        if varlen_pooled_list:
            varlen_pooled = torch.cat(varlen_pooled_list, dim=-1)
            feature_concat = torch.cat([dense_features, sparse_embedded, varlen_pooled], dim=-1)
        else:
            feature_concat = torch.cat([dense_features, sparse_embedded], dim=-1)
        
        feature_projected = self.feature_projection(feature_concat)
        
        # Fusion: combine text and feature embeddings
        fusion_embeddings = text_projected + feature_projected
        
        # Pass through fusion MLP layers
        for layer_norm, mlp_layer in zip(self.layer_norms, self.fusion_layers):
            residual = fusion_embeddings
            fusion_embeddings = layer_norm(fusion_embeddings)
            fusion_embeddings = mlp_layer(fusion_embeddings)
            fusion_embeddings = self.dropout(fusion_embeddings)
            fusion_embeddings = residual + fusion_embeddings
        
        fusion_embeddings = self.final_norm(fusion_embeddings)
        
        # Get predictions
        logits = self.regression_head(fusion_embeddings).squeeze(-1)
        
        # Calculate loss
        loss = None
        if labels is not None:
            loss = F.mse_loss(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )