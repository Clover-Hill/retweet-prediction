import pdb
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.integrations import use_kernel_forward_from_hub
# Qwen3MLP can't be imported directly
from transformers.activations import ACT2FN
from transformers import Qwen3Model, Qwen3Config, Qwen3PreTrainedModel

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
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class RetweetConfig(Qwen3Config):
    """Configuration class for Retweet Prediction models, extending Qwen3Config."""
    
    def __init__(
        self,
        mlp_num: int = 4,
        scalar_features_dim: int = 7,
        dropout_rate: float = 0.1,
        neg_pos_ratio: float = 1.0,
        num_class: int = 16,
        **kwargs
    ):
        """
        Args:
            hidden_size: Hidden dimension for fusion layer (will be split in half for text and scalar features)
            mlp_num: Number of Qwen3MLP layers to use after fusion
            scalar_features_dim: Number of scalar features (default: 7)
            dropout_rate: Dropout rate for regularization
            pos_neg_ratio: Positive to negative sample ratio for weighted BCE loss
            **kwargs: Additional arguments for Qwen3Config
        """
        super().__init__(**kwargs)
        self.mlp_num = mlp_num
        self.scalar_features_dim = scalar_features_dim
        self.dropout_rate = dropout_rate
        self.neg_pos_ratio = neg_pos_ratio
        self.num_class = num_class

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Extract embeddings from the last token position.
    
    Args:
        last_hidden_states: Hidden states from the model [batch_size, seq_len, hidden_size]
        attention_mask: Attention mask [batch_size, seq_len]
    
    Returns:
        Pooled embeddings [batch_size, hidden_size]
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class RetweetBaseModel(Qwen3PreTrainedModel):
    """Base model for retweet prediction with Qwen3 backbone."""
    
    config_class = RetweetConfig
    base_model_prefix = "qwen_model"
    
    # !!!YOU MUST explicitly define _init_weights, module.apply() is tricky
    # @torch.no_grad()
    # def initialize_weights(self):
    #     """
    #     This is equivalent to calling `self.apply(self._initialize_weights)`, but correctly handles composite models.
    #     This function dynamically dispatches the correct `init_weights` function to the modules as we advance in the
    #     module graph along the recursion. It can handle an arbitrary number of sub-models. Without it, every composite
    #     model would have to recurse a second time on all sub-models explicitly in the outer-most `_init_weights`, which
    #     is extremely error prone and inefficient.

    #     Note that the `torch.no_grad()` decorator is very important as well, as most of our `_init_weights` do not use
    #     `torch.nn.init` functions (which are all no_grad by default), but simply do in-place ops such as
    #     `module.weight.data.zero_()`.
    #     """
    #     if not hasattr(torch.nn.Module, "smart_apply"):
    #         # This function is equivalent to `torch.nn.Module.apply`, except that it dynamically adjust the function
    #         # to apply as we go down the graph
    #         def smart_apply(self, fn):
    #             for module in self.children():
    #                 # We found a sub-model: recursively dispatch its own init function now!
    #                 if isinstance(module, PreTrainedModel):
    #                     module.smart_apply(module._initialize_weights)
    #                 else:
    #                     module.smart_apply(fn)
    #             fn(self)
    #             return self

    #         torch.nn.Module.smart_apply = smart_apply

    #     # Let the magic happen with this simple call
    #     self.smart_apply(self._initialize_weights)
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

    def __init__(self, config: RetweetConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize Qwen3 model as the base
        self.qwen_model = Qwen3Model(config)
        
        # Projection layers for fusion
        self.text_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.scalar_projection = nn.Linear(config.scalar_features_dim, config.hidden_size)
        self.out_projection = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
        # MLP layers with layer norms (Pre-LN pattern)
        self.mlp_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        mlp_config = type('MLPConfig', (), {
            'hidden_size': config.hidden_size*2,
            'intermediate_size': config.hidden_size * 4,  # Standard MLP expansion ratio
            'hidden_act': config.hidden_act
        })()
        
        for _ in range(config.mlp_num):
            self.mlp_layers.append(Qwen3MLP(mlp_config))
            self.layer_norms.append(Qwen3RMSNorm(config.hidden_size * 2, eps=config.rms_norm_eps))
        
        # Final layer norm (optional but recommended)
        self.final_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        return self.qwen_model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.qwen_model.embed_tokens = value
    
    def forward_base(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        scalar_features: torch.FloatTensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Forward pass for base model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            scalar_features: Scalar features [batch_size, 7]
            Other args are passed to Qwen3Model
        
        Returns:
            Fusion embeddings after MLP layers [batch_size, hidden_size]
        """
        # Get Qwen3 outputs
        qwen_outputs = self.qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        # Get last hidden states and apply last token pooling
        last_hidden_states = qwen_outputs.last_hidden_state
        text_embeddings = last_token_pool(last_hidden_states, attention_mask)
        
        # Project text embeddings to hidden_size
        text_projected = self.text_projection(text_embeddings)
        
        # Project scalar features to hidden_size
        scalar_projected = self.scalar_projection(scalar_features)
        
        # Concatenate and normalize
        fusion_embeddings = torch.cat([text_projected, scalar_projected], dim=-1)
        
        # Pass through MLP layers with Pre-LN pattern
        for layer_norm, mlp_layer in zip(self.layer_norms, self.mlp_layers):
            # Pre-LN: Normalize -> MLP -> Residual
            residual = fusion_embeddings
            fusion_embeddings = layer_norm(fusion_embeddings)
            fusion_embeddings = mlp_layer(fusion_embeddings)
            fusion_embeddings = self.dropout(fusion_embeddings)  # Apply dropout
            fusion_embeddings = residual + fusion_embeddings
        
        # Final normalization
        fusion_embeddings = self.out_projection(fusion_embeddings)
        fusion_embeddings = self.final_norm(fusion_embeddings)
        
        return fusion_embeddings

class RetweetMultiRegressionModel(Qwen3PreTrainedModel):
    """
    Retweet viral class prediction model using Cross-Entropy loss.
    Based on RetweetRegressionModel but adapted for multi-class classification.
    """
    
    config_class = RetweetConfig
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

    def __init__(self, config: RetweetConfig):
        super().__init__(config)
        self.config = config
        
        # Number of classes for viral classification
        self.num_class = getattr(config, 'num_class', 16)  # Default to 16 classes
        
        # Initialize Qwen3 model as the base
        self.qwen_model = Qwen3Model(config)
        
        # MLP layers with layer norms (Pre-LN pattern)
        self.mlp_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        mlp_config = type('MLPConfig', (), {
            'hidden_size': config.hidden_size,
            'intermediate_size': config.hidden_size * 4,  # Standard MLP expansion ratio
            'hidden_act': config.hidden_act
        })()
        
        for _ in range(config.mlp_num):
            self.mlp_layers.append(Qwen3MLP(mlp_config))
            self.layer_norms.append(Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps))
        
        # Final layer norm
        self.final_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Classification head for multi-class prediction
        self.classification_head = nn.Linear(config.hidden_size, self.num_class)
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        return self.qwen_model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.qwen_model.embed_tokens = value
    
    def set_class_weights(self, weights):
        """Set class weights for handling imbalanced classes"""
        self.class_weights = torch.tensor(weights, dtype=torch.float32)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> SequenceClassifierOutput:
        """
        Forward pass for multi-class classification model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Class labels [batch_size] with values in [0, num_class)
            Other args are passed to Qwen3Model
        
        Returns:
            SequenceClassifierOutput with loss and logits
        """
        # Get Qwen3 outputs
        qwen_outputs = self.qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        # Get last hidden states and apply last token pooling
        last_hidden_states = qwen_outputs.last_hidden_state
        text_embeddings = last_token_pool(last_hidden_states, attention_mask)
        
        fusion_embeddings = text_embeddings
        
        # Pass through MLP layers with Pre-LN pattern
        for layer_norm, mlp_layer in zip(self.layer_norms, self.mlp_layers):
            # Pre-LN: Normalize -> MLP -> Residual
            residual = fusion_embeddings
            fusion_embeddings = layer_norm(fusion_embeddings)
            fusion_embeddings = mlp_layer(fusion_embeddings)
            fusion_embeddings = residual + fusion_embeddings
        
        # Final normalization
        fusion_embeddings = self.final_norm(fusion_embeddings)

        # Apply classification head
        logits = self.classification_head(fusion_embeddings)  # [batch_size, num_class]
        
        loss = None
        if labels is not None:
            # Cross-entropy loss with optional class weights
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=qwen_outputs.hidden_states if output_hidden_states else None,
            attentions=qwen_outputs.attentions if output_attentions else None,
        )

class RetweetRegressionModel(Qwen3PreTrainedModel):
    """Base model for retweet prediction with Qwen3 backbone."""
    
    config_class = RetweetConfig
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

    def __init__(self, config: RetweetConfig):
        super().__init__(config)
        self.config = config
        
        # Initialize Qwen3 model as the base
        self.qwen_model = Qwen3Model(config)
        
        # MLP layers with layer norms (Pre-LN pattern)
        self.mlp_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        mlp_config = type('MLPConfig', (), {
            'hidden_size': config.hidden_size,
            'intermediate_size': config.hidden_size * 4,  # Standard MLP expansion ratio
            'hidden_act': config.hidden_act
        })()
        
        for _ in range(config.mlp_num):
            self.mlp_layers.append(Qwen3MLP(mlp_config))
            self.layer_norms.append(Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps))
        
        # Final layer norm (optional but recommended)
        self.final_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout_rate)

        # Output layer for regression
        self.regression_head = nn.Linear(config.hidden_size, 1)
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        return self.qwen_model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.qwen_model.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for base model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            scalar_features: Scalar features [batch_size, 7]
            Other args are passed to Qwen3Model
        
        Returns:
            Fusion embeddings after MLP layers [batch_size, hidden_size]
        """
        # Get Qwen3 outputs
        qwen_outputs = self.qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        # Get last hidden states and apply last token pooling
        last_hidden_states = qwen_outputs.last_hidden_state
        text_embeddings = last_token_pool(last_hidden_states, attention_mask)
        
        fusion_embeddings = text_embeddings
        
        # Pass through MLP layers with Pre-LN pattern
        for layer_norm, mlp_layer in zip(self.layer_norms, self.mlp_layers):
            # Pre-LN: Normalize -> MLP -> Residual
            residual = fusion_embeddings
            fusion_embeddings = layer_norm(fusion_embeddings)
            fusion_embeddings = mlp_layer(fusion_embeddings)
            fusion_embeddings = self.dropout(fusion_embeddings)  # Apply dropout
            fusion_embeddings = residual + fusion_embeddings
        
        # Final normalization
        fusion_embeddings = self.final_norm(fusion_embeddings)

        # Apply regression head
        logits = self.regression_head(fusion_embeddings).squeeze(-1)  # [batch_size]
        
        loss = None
        if labels is not None:
            # L1 loss for regression
            loss_fct = nn.L1Loss()
            loss = loss_fct(logits, labels.float())
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

# class RetweetRegressionModelLegacy(RetweetBaseModel):
#     """Model for retweet count regression."""
    
#     def __init__(self, config: RetweetConfig):
#         super().__init__(config)
        
#         # Output layer for regression
#         self.regression_head = nn.Linear(config.hidden_size, 1)
        
#         # Initialize weights
#         self.post_init()
    
#     def forward(
#         self,
#         input_ids: torch.LongTensor,
#         attention_mask: torch.Tensor,
#         scalar_features: torch.FloatTensor,
#         labels: Optional[torch.FloatTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, SequenceClassifierOutput]:
#         """
#         Forward pass for regression model.
        
#         Args:
#             input_ids: Input token IDs [batch_size, seq_len]
#             attention_mask: Attention mask [batch_size, seq_len]
#             scalar_features: Scalar features [batch_size, 7]
#             labels: Retweet counts for L1 loss [batch_size]
#             Other args are passed to Qwen3Model
        
#         Returns:
#             SequenceClassifierOutput with logits and optional loss
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
#         # Get fusion embeddings from base model
#         fusion_embeddings = self.forward_base(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             scalar_features=scalar_features,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict
#         )
        
#         # Apply regression head
#         logits = self.regression_head(fusion_embeddings).squeeze(-1)  # [batch_size]
        
#         loss = None
#         if labels is not None:
#             # L1 loss for regression
#             loss_fct = nn.L1Loss()
#             loss = loss_fct(logits, labels.float())
        
#         if not return_dict:
#             output = (logits,)
#             return ((loss,) + output) if loss is not None else output
        
#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=None,
#             attentions=None,
#         )


class RetweetClassificationModel(RetweetBaseModel):
    """Model for viral tweet classification."""
    
    def __init__(self, config: RetweetConfig):
        super().__init__(config)
        
        # Output layer for binary classification (single output for BCE)
        self.classification_head = nn.Linear(config.hidden_size, 1)
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        scalar_features: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        """
        Forward pass for classification model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            scalar_features: Scalar features [batch_size, 7]
            labels: Binary labels for if_viral (0 or 1) [batch_size]
            Other args are passed to Qwen3Model
        
        Returns:
            SequenceClassifierOutput with logits and optional loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get fusion embeddings from base model
        fusion_embeddings = self.forward_base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            scalar_features=scalar_features,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        # Apply classification head
        logits = self.classification_head(fusion_embeddings).squeeze(-1)  # [batch_size]
        
        loss = None
        if labels is not None:
            # Calculate class weights based on pos_neg_ratio
            # pos_weight = number of negative samples / number of positive samples
            # If pos_neg_ratio = 0.1, it means pos:neg = 1:10, so pos_weight = 10
            pos_weight = torch.tensor([self.config.neg_pos_ratio], device=logits.device)
            
            # BCEWithLogitsLoss with pos_weight for handling imbalanced data
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fct(logits, labels.float())
        
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

# Example usage:
if __name__ == "__main__":
    # Create configuration
    config = RetweetConfig.from_pretrained("/fs-computility/plm/shared/jqcao/models/Qwen3/Qwen3-Embedding-0.6B")
    print(config)
    
    # Load pretrained Qwen3 weights
    regression_model = RetweetRegressionModel.from_pretrained(
        "/fs-computility/plm/shared/jqcao/models/Qwen3/Qwen3-Embedding-0.6B",
        config=config,
    )
    
    # Example forward pass
    # batch_size = 2
    # seq_len = 128
    
    # input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    # attention_mask = torch.ones(batch_size, seq_len)
    # scalar_features = torch.randn(batch_size, 7)
    
    # # Regression
    # retweet_counts = torch.randint(0, 1000, (batch_size,)).float()
    # reg_outputs = regression_model(
    #     input_ids=input_ids,
    #     attention_mask=attention_mask,
    #     scalar_features=scalar_features,
    #     labels=retweet_counts
    # )
    # print(f"Regression loss: {reg_outputs.loss}")
    # print(f"Predicted retweet counts: {reg_outputs.logits}")
    
    # # Classification
    # if_viral = torch.randint(0, 2, (batch_size,))  # Binary labels: 0 or 1
    # clf_outputs = classification_model(
    #     input_ids=input_ids,
    #     attention_mask=attention_mask,
    #     scalar_features=scalar_features,
    #     labels=if_viral
    # )
    # print(f"Classification loss: {clf_outputs.loss}")
    # print(f"Viral prediction logits: {clf_outputs.logits}")
    # print(f"Viral prediction probabilities: {torch.sigmoid(clf_outputs.logits)}")