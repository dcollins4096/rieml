import torch

# Constants
gamma = 1.6667  # Ideal gas gamma

def primitive_to_characteristic(prim):
    """Convert primitive variables to characteristic variables.

    Args:
        prim (tensor): [..., 3] with [..., rho, v, p]

    Returns:
        w (tensor): [..., 3] with characteristic variables [w1, w2, w3]
    """
    rho, v, p = prim[0], prim[2], prim[ 1]
    c = torch.sqrt(gamma * p / rho)

    delta_rho = prim[ 0]
    delta_v = prim[1]
    delta_p = prim[2]

    w1 = delta_v - delta_p / (rho * c)
    w2 = delta_rho - delta_p / (c ** 2)
    w3 = delta_v + delta_p / (rho * c)
    #kludge
    w1 = rho
    w2 = p
    w3 = v

    return torch.stack([w1, w2, w3])

def characteristic_to_primitive(w, rho, p):
    """Convert characteristic variables back to primitive.

    Args:
        w (tensor): [..., 3] with [w1, w2, w3]
        rho (tensor): [...] original rho
        p (tensor): [...] original p

    Returns:
        prim (tensor): [..., 3] with [rho, v, p]
    """
    c = torch.sqrt(gamma * p / rho)

    w1, w2, w3 = w[ 0], w[ 1], w[ 2]

    delta_rho = w2
    delta_v = 0.5 * (w1 + w3)
    delta_p = 0.5 * rho * c * (w3 - w1)

    delta_rho = w1
    delta_p = w2
    delta_v = w3


    return torch.stack([delta_rho,delta_p , delta_v])

# Example wrapper
def nn_with_characteristics(nn_model, input_prim):
    """
    Wraps the NN so it operates in characteristic space.

    Args:
        nn_model (torch.nn.Module): a model that maps [..., 3] -> [..., 3]
        input_prim (tensor): [..., 3] with [rho, v, p]

    Returns:
        output_prim (tensor): [..., 3] predicted [rho, v, p]
    """
    rho = input_prim[..., 0]
    p   = input_prim[..., 2]

    w_in = primitive_to_characteristic(input_prim)
    w_out = nn_model(w_in)
    output_prim = characteristic_to_primitive(w_out, rho, p)
    return output_prim

