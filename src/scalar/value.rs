use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct Value(Rc<RefCell<Node>>);

#[derive(Clone, Debug)]
struct Node {
    data: f64,
    grad: f64,
    op: Op,
    prev: Vec<Value>,
}

#[derive(Clone, Debug)]
enum Op {
    Leaf,
    Add,
    Mul,
    Pow(f64),
    Exp,
    Log,
    Relu,
}

impl Value {
    pub fn new(data: f64) -> Self {
        Self(Rc::new(RefCell::new(Node {
            data,
            grad: 0.0,
            op: Op::Leaf,
            prev: Vec::new(),
        })))
    }

    fn from_op(data: f64, op: Op, prev: Vec<Value>) -> Self {
        Self(Rc::new(RefCell::new(Node {
            data,
            grad: 0.0,
            op,
            prev,
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn set_data(&self, data: f64) {
        self.0.borrow_mut().data = data;
    }

    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad = 0.0;
    }

    pub fn add(&self, other: &Value) -> Value {
        Self::from_op(
            self.data() + other.data(),
            Op::Add,
            vec![self.clone(), other.clone()],
        )
    }

    pub fn mul(&self, other: &Value) -> Value {
        Self::from_op(
            self.data() * other.data(),
            Op::Mul,
            vec![self.clone(), other.clone()],
        )
    }

    pub fn powf(&self, exponent: f64) -> Value {
        Self::from_op(
            self.data().powf(exponent),
            Op::Pow(exponent),
            vec![self.clone()],
        )
    }

    pub fn exp(&self) -> Value {
        Self::from_op(self.data().exp(), Op::Exp, vec![self.clone()])
    }

    pub fn log(&self) -> Value {
        Self::from_op(self.data().ln(), Op::Log, vec![self.clone()])
    }

    pub fn relu(&self) -> Value {
        Self::from_op(self.data().max(0.0), Op::Relu, vec![self.clone()])
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        self.build_topo(&mut visited, &mut topo);

        for node in &topo {
            node.0.borrow_mut().grad = 0.0;
        }
        self.0.borrow_mut().grad = 1.0;

        for node in topo.into_iter().rev() {
            node.propagate();
        }
    }

    fn build_topo(&self, visited: &mut HashSet<usize>, topo: &mut Vec<Value>) {
        let id = self.node_id();
        if !visited.insert(id) {
            return;
        }

        let prev = self.0.borrow().prev.clone();
        for child in prev {
            child.build_topo(visited, topo);
        }
        topo.push(self.clone());
    }

    fn propagate(&self) {
        let (op, grad, data, prev) = {
            let node = self.0.borrow();
            (node.op.clone(), node.grad, node.data, node.prev.clone())
        };

        match op {
            Op::Leaf => {}
            Op::Add => {
                add_grad(&prev[0], grad);
                add_grad(&prev[1], grad);
            }
            Op::Mul => {
                let lhs = prev[0].data();
                let rhs = prev[1].data();
                add_grad(&prev[0], rhs * grad);
                add_grad(&prev[1], lhs * grad);
            }
            Op::Pow(exponent) => {
                let base = prev[0].data();
                add_grad(&prev[0], exponent * base.powf(exponent - 1.0) * grad);
            }
            Op::Exp => {
                add_grad(&prev[0], data * grad);
            }
            Op::Log => {
                let input = prev[0].data();
                add_grad(&prev[0], grad / input);
            }
            Op::Relu => {
                let input = prev[0].data();
                if input > 0.0 {
                    add_grad(&prev[0], grad);
                }
            }
        }
    }

    fn node_id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }
}

fn add_grad(value: &Value, delta: f64) {
    value.0.borrow_mut().grad += delta;
}

#[cfg(test)]
mod tests {
    use super::Value;

    fn assert_close(lhs: f64, rhs: f64) {
        let eps = 1e-10;
        assert!(
            (lhs - rhs).abs() <= eps,
            "values differ: left={lhs}, right={rhs}"
        );
    }

    #[test]
    fn gradients_through_add_mul_relu() {
        let x = Value::new(2.0);
        let y = Value::new(-3.0);
        let z = Value::new(10.0);

        let q = x.mul(&y).add(&z);
        let out = q.relu();
        out.backward();

        assert_close(out.data(), 4.0);
        assert_close(x.grad(), -3.0);
        assert_close(y.grad(), 2.0);
        assert_close(z.grad(), 1.0);
    }

    #[test]
    fn gradients_through_pow_exp_log_chain() {
        let x = Value::new(1.5);

        let out = x.powf(2.0).exp().log();
        out.backward();

        assert_close(out.data(), 2.25);
        assert_close(x.grad(), 3.0);
    }

    #[test]
    fn relu_blocks_negative_gradient() {
        let x = Value::new(-2.0);
        let out = x.relu();
        out.backward();

        assert_close(out.data(), 0.0);
        assert_close(x.grad(), 0.0);
    }

    #[test]
    fn zero_grad_resets_leaf_gradients() {
        let x = Value::new(4.0);
        let squared = x.powf(2.0);
        squared.backward();
        assert_close(x.grad(), 8.0);

        x.zero_grad();
        assert_close(x.grad(), 0.0);

        let two = Value::new(2.0);
        let doubled = x.mul(&two);
        doubled.backward();
        assert_close(x.grad(), 2.0);
    }
}
