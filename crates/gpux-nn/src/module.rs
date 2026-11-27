use gpux_tensor::Tensor;

pub trait Module {
    fn forward(&self, input: &Tensor) -> gpux_core::Result<Tensor>;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameter_count(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

pub struct ParamGrad {
    pub param: Tensor,
    pub grad: Tensor,
}
