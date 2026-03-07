import torch


@torch.no_grad()
def token_preprocess(activation_a: torch.Tensor, activation_b: torch.Tensor):
    """
    Preprocess the activations for transport weight calculation.
    This function normalizes the activations and computes the mean and variance.

    Tokens Pre-Processing. To align the input activations of the two models, we first need to ensure that they have the same sequence length. If LA < LB, we interpolate the sequence length LA to match LB using bilinear interpolation. Then, we flatten the first two dimensions of the input activations; let M = N x LB be the total number of tokens after interpolation. The input activations can then be represented as: Hin,A ∈ RMxdin,A and Hin,B ∈ RMxdin,B , and similarly for the output activations.

    Args:
        activation_a (torch.Tensor): Activations from model A.
        activation_b (torch.Tensor): Activations from model B.
    """
    assert activation_a.dim() == 3, "Activation A must be a 3D tensor (N, LA, din_A)"
    assert activation_b.dim() == 3, "Activation B must be a 3D tensor (N, LB, din_B)"
    assert (
        activation_a.shape[0] == activation_b.shape[0]
    ), "Batch size (N) must be the same for both activations"

    # Ensure the sequence lengths match
    if activation_a.shape[1] != activation_b.shape[1]:
        # TODO: The implementation of interpolation might be wrong, need to verify it when the original code of THESEUS is available.
        # Interpolate activation_a to match the sequence length of activation_b.
        # bilinear mode requires 4D input (N, C, H, W), so we add a dummy H=1 dimension.
        activation_a = (
            torch.nn.functional.interpolate(
                activation_a.permute(0, 2, 1).unsqueeze(2),  # (N, din_A, 1, LA)
                size=(1, activation_b.shape[1]),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(2)
            .permute(0, 2, 1)
        )  # (N, LB, din_A)

    # Flatten the first two dimensions
    M = activation_a.shape[0] * activation_a.shape[1]
    din_a = activation_a.shape[2]
    din_b = activation_b.shape[2]

    hin_a = activation_a.view(M, din_a)
    hin_b = activation_b.view(M, din_b)

    return hin_a, hin_b
