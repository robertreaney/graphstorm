import logging
import torch
import torch.nn as nn
from torch.nn import Linear, LayerNorm, MultiheadAttention, ModuleList, Sequential, Module, ReLU, GELU
from torch.nn.functional import relu, dropout, gelu
from torch.distributions.normal import Normal
import numpy as np

import structlog as sl

logger = sl.get_logger(__name__)

# HMA
class AttentionBlock(Module):
    def __init__(self, dim_model, dim_key, dim_value, dim_ffn, attention_heads, dropout_rate_attention=.05, **kwargs):
        super(AttentionBlock, self).__init__()
        self.dim_model = dim_model
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dim_ffn = dim_ffn
        self.attention_heads = attention_heads
        self.dropout_rate_attention = dropout_rate_attention

        self.q_proj = Linear(self.dim_model, self.attention_heads * self.dim_key, bias=False)  # 4 → 56
        self.k_proj = Linear(self.dim_model, self.attention_heads * self.dim_key, bias=False)  # 4 → 56
        self.v_proj = Linear(self.dim_model, self.attention_heads * self.dim_value, bias=False)  # 4 → 21

        self.mha = MultiheadAttention(
            # embed_dim=self.dim_model * self.params['attention_heads'], # 4 * 8 = 32
            embed_dim=self.attention_heads * self.dim_key,  # heads * query_dim
            num_heads=self.attention_heads,
            dropout=self.dropout_rate_attention,
            bias=False,
            batch_first=True,
            kdim=self.attention_heads * self.dim_key,
            vdim=self.attention_heads * self.dim_value
        )

        self.out_proj = Linear(self.attention_heads * self.dim_key, self.dim_model, bias=False)  # 21 → 4

        # these values need to be changed
        self.norm = LayerNorm(self.dim_model, eps=1e-6)

        self.ffn = Sequential(
            Linear(self.dim_model, out_features=self.dim_ffn, bias=False),
            ReLU(),
            Linear(in_features=self.dim_ffn, out_features=self.dim_model, bias=False)
        )

        self.out_norm = LayerNorm(self.dim_model, eps=1e-6)

    def forward(self, x, edge_index=None):
        # multi-head attention
        q = self.q_proj(x)  
        k = self.k_proj(x)
        v = self.v_proj(x)
        # q = k = v = x

        attn_output, _ = self.mha(q, k, v)

        attn_output = self.out_proj(attn_output)  # (batch, 128, 4)

        # residual connection
        x = x + attn_output
        
        x = self.norm(x)

        # ffn
        ffn_output = self.ffn(x)

        # residual
        x = x + ffn_output

        # layernorm
        x = self.out_norm(x)

        return x

class HMA(Module): 
    def __init__(
        self,
        *,
        in_features: int = 512, 
        out_features: int = 18767, #hma 19
        activation_input: str = "linear",
        attention_heads: int = 4,
        dim_ffn: int = 648,
        dim_input: int = 128,
        dim_key: int = 5,
        dim_model: int = 14,
        dim_value: int = 8,
        dropout_rate_attention: float = 0.45,
        dropout_rate_bottleneck: float = 0.0,
        dropout_rate_input: float = 0.25,
        gaussian_noise_bottleneck: float = 0.0,
        gaussian_noise_input: float = 0.0,
        n_attention_steps: int = 5,
        **kwargs
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_input = activation_input
        self.attention_heads = attention_heads
        self.dim_ffn = dim_ffn
        self.dim_input = dim_input
        self.dim_key = dim_key
        self.dim_model = dim_model
        self.dim_value = dim_value
        self.dropout_rate_attention = dropout_rate_attention
        self.dropout_rate_bottleneck = dropout_rate_bottleneck
        self.dropout_rate_input = dropout_rate_input
        self.gaussian_noise_bottleneck = gaussian_noise_bottleneck
        self.gaussian_noise_input = gaussian_noise_input
        self.n_attention_steps = n_attention_steps
        self.input_dimension = self.dim_model * self.dim_input # 14 * 128


        # encoder
        # add activation
        self.activation_input
        self.input_proj = Linear(in_features, self.input_dimension)
        # # transformations
        self.attention_layers = ModuleList([
            AttentionBlock(
                dim_model=self.dim_model, 
                dim_key=self.dim_key, 
                dim_value=self.dim_value, 
                dim_ffn=self.dim_ffn,
                attention_heads=self.attention_heads, 
                dropout_rate=self.dropout_rate_attention,
            ) for _ in range(self.n_attention_steps)
        ])

        # decoder
        self.decoder = Linear(self.input_dimension, out_features=self.out_features)

    def forward(self, x, *args, **kwargs):
        x = self.input_proj(x)

        x = dropout(x, p=self.dropout_rate_input, training=self.training) # (batch_size, 512)

        # # TRANSFORMATIONS
        # # reshape to have dim_input "embeddings" with "dim_model" dimensions
        x = x.view(-1, self.dim_input, self.dim_model) # (batch_size, 128, 4)

        # # normalize by sqrt(dim_model) as in the paper
        x = x * (self.dim_model ** .5)

        # # pass through attn blocks
        for attn in self.attention_layers:
            x = attn(x)

        # # OUTPUT
        x = x.flatten(start_dim=1)  # flatten with respect of batch
        
        # x = dropout(x, p=self.hyperparameters['dropout_rate_bottleneck'], training=self.training)  # this is 0
        # x = x + torch.randn_like(x) * 0  # this hyperparam is also 0
        
        return self.decoder(x)
    
# Mixture of Experts
# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
# https://github.com/davidmrau/mixture-of-experts/blob/master/moe.py
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out


class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """
    def __init__(self, input_size, output_size, num_experts=16, hidden_size=1024, noisy_gating=True, k=4, hidden_size_factor=None, loss_coef=1e-2, **kwargs):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.loss_coef = loss_coef
        if hidden_size_factor is not None:
            self.hidden_size = int(hidden_size_factor * input_size)
            logging.warning("MoE node encoder arg hidden_size_factor is not None. Default will ignore hidden_size and use this instead.")
            
        # instantiate experts
        self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        
        if self.training:
            return y, loss
        return y
