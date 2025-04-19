import argparse
import os
import torch
import pandas as pd
from model_prot2motif import EncoderRNN, AttnDecoderRNN, Seq2Seq
from train_model import train


def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(description='prot2motif Model Training Pipeline')
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to training dataset CSV file')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Hidden layer size for encoder/decoder (default: 128)')
    parser.add_argument('--input_size', type=int, default=337,
                       help='Input feature size for encoder (default: 337)')
    parser.add_argument('--output_size', type=int, default=5,
                       help='Output size for decoder (default: 5)')
    parser.add_argument('--batch_limit', type=int, nargs=2, default=[None, None],
                       help='Limit training/validation batches')
    
    return parser.parse_args()

def initialize_model(args, device):
    """Initialize prot2motif model components"""
    encoder = EncoderRNN(args.input_size, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, args.output_size).to(device)
    return Seq2Seq(encoder, decoder).to(device)

def load_and_split_data(dataset_path):
    """Load dataset and perform stratified split
    
    Parameters
    ----------
    dataset_path : the path to the dataset

    Returns
    -------
    A tuple of a split dataset into training, validation, and test samples
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file {dataset_path} not found")
    
    dataset = pd.read_csv(dataset_path)
    
    # Stratified split proportions
    n = sorted_dataset.shape[0]
    return (
        sorted_dataset.iloc[:int(n * 0.8)],  # Train samples
        sorted_dataset.iloc[int(n * 0.8):int(n * 0.9)],  # Validation samples
        sorted_dataset.iloc[int(n * 0.9):]  # Test samples
    )

def main():
    """Main training pipeline execution"""
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\nInitializing Training Pipeline")
    print(f"Using computation device: {device}")
    
    try:
        train_data, val_data, test_data = load_and_split_data(args.dataset)
        print("\nDataset Statistics:")
        print(f"  Training samples: {len(train_data):,}")
        print(f"  Validation samples: {len(val_data):,}")
        print(f"  Test samples: {len(test_data):,}")
        
        model = initialize_model(args, device)
        print("\nModel Architecture:")
        print(f"  Encoder: Input={args.input_size}, Hidden={args.hidden_size}")
        print(f"  Decoder: Hidden={args.hidden_size}, Output={args.output_size}")
        
        print("\nStarting Training...")
        train(
            train_samples=train_data[:args.batch_limit[0]] if args.batch_limit[0] else train_data,
            val_samples=val_data[:args.batch_limit[1]] if args.batch_limit[1] else val_data,
            model=model,
            n_epochs=args.epochs
        )
        
    except Exception as e:
        print(f"\nCritical Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
