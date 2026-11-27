use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn scalar() -> Self {
        Self { dims: vec![1] }
    }

    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn size(&self, dim: usize) -> usize {
        self.dims[dim]
    }

    pub fn last_dim(&self) -> usize {
        *self.dims.last().unwrap_or(&1)
    }

    pub fn byte_size(&self) -> usize {
        self.numel() * std::mem::size_of::<f32>()
    }

    pub fn rows(&self) -> usize {
        if self.dims.len() < 2 {
            1
        } else {
            self.dims[..self.dims.len() - 1].iter().product()
        }
    }

    pub fn cols(&self) -> usize {
        self.last_dim()
    }

    pub fn transpose(&self) -> Self {
        assert!(self.ndim() == 2, "transpose only for 2D tensors");
        Self::new(vec![self.dims[1], self.dims[0]])
    }

    pub fn matmul_shape(&self, other: &Shape) -> Shape {
        let m = if self.ndim() >= 2 {
            self.dims[self.ndim() - 2]
        } else {
            1
        };
        let n = other.last_dim();
        if self.ndim() <= 2 && other.ndim() <= 2 {
            Shape::new(vec![m, n])
        } else {
            let mut batch_dims: Vec<usize> = self.dims[..self.ndim() - 2].to_vec();
            batch_dims.push(m);
            batch_dims.push(n);
            Shape::new(batch_dims)
        }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims.to_vec())
    }
}

macro_rules! impl_shape_from_array {
    ($($n:expr),*) => {
        $(
            impl From<[usize; $n]> for Shape {
                fn from(dims: [usize; $n]) -> Self {
                    Self::new(dims.to_vec())
                }
            }
        )*
    };
}

impl_shape_from_array!(1, 2, 3, 4, 5);
