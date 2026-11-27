use std::sync::Arc;

use crate::tensor::Tensor;

#[derive(Clone)]
pub enum GradFn {
    None,
    Add(Arc<TensorNode>, Arc<TensorNode>),
    Sub(Arc<TensorNode>, Arc<TensorNode>),
    Mul(Arc<TensorNode>, Arc<TensorNode>),
    MulScalar(Arc<TensorNode>, f32),
    MatMul(Arc<TensorNode>, Arc<TensorNode>),
    Relu(Arc<TensorNode>),
    Gelu(Arc<TensorNode>),
    Softmax(Arc<TensorNode>, Tensor),
    LayerNorm {
        input: Arc<TensorNode>,
        gamma: Arc<TensorNode>,
        mean: Tensor,
        rstd: Tensor,
    },
    Embedding(Arc<TensorNode>, Tensor),
    CrossEntropy {
        logits_node: Arc<TensorNode>,
        softmax_out: Tensor,
        targets: Tensor,
    },
    BroadcastAddBias(Arc<TensorNode>, Arc<TensorNode>),
    Transpose(Arc<TensorNode>),
    Reshape(Arc<TensorNode>),
    Sum(Arc<TensorNode>),
    AddPositionEmbedding {
        input: Arc<TensorNode>,
        pos_weight: Arc<TensorNode>,
    },
    AttentionScores {
        q: Arc<TensorNode>,
        k: Arc<TensorNode>,
        scale: f32,
    },
    AttentionWeightedValues {
        weights_node: Arc<TensorNode>,
        v: Arc<TensorNode>,
    },
    ReshapeForAttention(Arc<TensorNode>),
    ReshapeFromAttention(Arc<TensorNode>),
}

pub struct TensorNode {
    pub tensor: Tensor,
    pub grad_fn: GradFn,
    pub requires_grad: bool,
}

impl TensorNode {
    pub fn leaf(tensor: Tensor, requires_grad: bool) -> Arc<Self> {
        Arc::new(Self {
            tensor,
            grad_fn: GradFn::None,
            requires_grad,
        })
    }

    pub fn with_grad_fn(tensor: Tensor, grad_fn: GradFn) -> Arc<Self> {
        Arc::new(Self {
            tensor,
            grad_fn,
            requires_grad: true,
        })
    }
}

pub struct GradientTape {
    nodes: Vec<(Arc<TensorNode>, Tensor)>,
}

impl GradientTape {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn accumulate(&mut self, node: &Arc<TensorNode>, grad: Tensor) {
        self.nodes.push((node.clone(), grad));
    }

    pub fn grads(&self) -> &[(Arc<TensorNode>, Tensor)] {
        &self.nodes
    }
}
