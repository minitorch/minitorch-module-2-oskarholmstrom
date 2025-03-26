from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
   # Create a list from the input values to allow modification
    vals_list = list(vals)
    
    # Create values for f(x + epsilon) calculation
    vals_plus = vals_list.copy()
    vals_plus[arg] = vals_plus[arg] + epsilon
    
    # Create values for f(x - epsilon) calculation
    vals_minus = vals_list.copy()
    vals_minus[arg] = vals_minus[arg] - epsilon
    
    # Apply central difference formula: [f(x + epsilon) - f(x - epsilon)] / (2 * epsilon)
    return (f(*vals_plus) - f(*vals_minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # Dictionary to track visited variables
    visited = {}
    
    # List to store variables in topological order
    topo_order = []
    
    def visit(var: Variable) -> None:
        # Skip if variable is constant or already visited
        if var.is_constant() or var.unique_id in visited:
            return
            
        # Mark as visited with a temporary flag
        visited[var.unique_id] = True
        
        # Visit all parents (dependencies) first
        if hasattr(var, 'parents'):
            for parent in var.parents:
                visit(parent)
        
        # Add this variable to the order after all its dependencies
        topo_order.append(var)
    
    # Start the traversal from the rightmost variable
    visit(variable)
    
    # Return in reverse order (from rightmost/output to leftmost/input)
    return topo_order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
     # Dictionary to keep track of variables we've processed
    visited = {}
    
    # Dictionary to store derivatives for each variable
    derivatives = {}
    
    # Start with the output variable and its derivative
    derivatives[variable.unique_id] = deriv
    
    # Create a queue of variables to process
    queue = [variable]
    
    while queue:
        var = queue.pop(0)
        
        # Skip if already visited or is a constant
        if var.unique_id in visited or var.is_constant():
            continue
        
        visited[var.unique_id] = True
        d_output = derivatives[var.unique_id]
        
        # If it's a leaf variable, accumulate the derivative
        if var.is_leaf():
            var.accumulate_derivative(d_output)
        # Otherwise, propagate to its parents
        elif not var.is_constant():
            # Get parent variables and their gradients
            for parent_var, grad in var.chain_rule(d_output):
                # Add parent to queue if not already processed
                if parent_var.unique_id not in visited:
                    queue.append(parent_var)
                
                # Add/update gradient for parent
                if parent_var.unique_id in derivatives:
                    derivatives[parent_var.unique_id] += grad
                else:
                    derivatives[parent_var.unique_id] = grad

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
