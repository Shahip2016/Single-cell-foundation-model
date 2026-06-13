"""
Model export utilities for CellFM.
"""
import torch
import os

def export_to_onnx(model, output_path: str, n_genes: int, device: str = "cpu"):
    """
    Export the CellFM model to ONNX format.
    
    Args:
        model: The CellFM model instance
        output_path: Path to save the .onnx file
        n_genes: Number of genes the model expects
        device: Device to use for dummy inputs
    """
    model.eval()
    model.to(device)
    
    # Create dummy inputs for tracing
    # (batch_size, seq_len)
    dummy_gene_ids = torch.randint(0, n_genes, (1, 2048), device=device)
    dummy_gene_values = torch.randn(1, 2048, device=device)
    
    print(f"Exporting model to {output_path}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    torch.onnx.export(
        model, 
        (dummy_gene_ids, dummy_gene_values), 
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['gene_ids', 'gene_values'],
        output_names=['output'],
        dynamic_axes={
            'gene_ids': {0: 'batch_size', 1: 'seq_len'},
            'gene_values': {0: 'batch_size', 1: 'seq_len'},
            'output': {0: 'batch_size', 1: 'seq_len'}
        }
    )
    print("Export complete.")
    return output_path
