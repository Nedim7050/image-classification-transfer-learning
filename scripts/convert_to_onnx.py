"""
Convert PyTorch model checkpoint to ONNX format for lightweight deployment
"""

import argparse
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.infer import load_model


def convert_to_onnx(model_path: str, output_path: str, img_size: int = 224, device: str = 'cpu'):
    """
    Convert PyTorch model to ONNX format
    
    Args:
        model_path: Path to PyTorch checkpoint (.pth)
        output_path: Path to save ONNX model (.onnx)
        img_size: Input image size
        device: Device to use for conversion
    """
    print(f"Loading model from {model_path}...")
    model, class_names, model_name = load_model(model_path, device)
    
    print(f"Model: {model_name}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Input size: {img_size}x{img_size}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Export to ONNX
    print(f"\nExporting to ONNX format...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
        
        print(f"✅ Model successfully exported to {output_path}")
        
        # Get file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"📦 ONNX model size: {file_size:.2f} MB")
        
        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model verification passed")
        except ImportError:
            print("⚠️  onnx package not installed, skipping verification")
        except Exception as e:
            print(f"⚠️  ONNX verification warning: {e}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error during ONNX export: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_onnx_model(onnx_path: str, img_size: int = 224):
    """
    Test ONNX model with dummy input
    
    Args:
        onnx_path: Path to ONNX model
        img_size: Input image size
    """
    try:
        import onnxruntime as ort
        import numpy as np
        
        print(f"\nTesting ONNX model...")
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path)
        
        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print(f"Input name: {input_name}")
        print(f"Output name: {output_name}")
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        
        # Run inference
        outputs = session.run([output_name], {input_name: dummy_input})
        
        print(f"✅ ONNX model test successful")
        print(f"Output shape: {outputs[0].shape}")
        
        return True
    
    except ImportError:
        print("⚠️  onnxruntime not installed, skipping ONNX test")
        return False
    except Exception as e:
        print(f"❌ Error testing ONNX model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX format')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to PyTorch checkpoint (.pth)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save ONNX model (.onnx). Default: same as model_path with .onnx extension')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use for conversion')
    parser.add_argument('--test', action='store_true',
                       help='Test ONNX model after conversion')
    
    args = parser.parse_args()
    
    # Set output path
    if args.output_path is None:
        output_path = args.model_path.replace('.pth', '.onnx')
    else:
        output_path = args.output_path
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Convert to ONNX
    success = convert_to_onnx(args.model_path, output_path, args.img_size, args.device)
    
    if success and args.test:
        import numpy as np
        test_onnx_model(output_path, args.img_size)
    
    print(f"\n{'='*60}")
    print("Conversion completed!")
    print(f"ONNX model saved to: {output_path}")
    print(f"\nTo use the ONNX model:")
    print(f"  import onnxruntime as ort")
    print(f"  session = ort.InferenceSession('{output_path}')")
    print(f"  outputs = session.run(['output'], {{'input': input_array}})")


if __name__ == '__main__':
    main()

