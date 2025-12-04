import pandas as pd
from sklearn.model_selection import train_test_split

# Function to split the data
def split_data(train_percent, valid_percent):
    input_file = "fake reviews dataset.csv"
    output_train = "train_data.csv"
    output_valid = "val_data.csv"
    output_test = "test_data.csv"
    
    # Load the dataset
    df = pd.read_csv(input_file, encoding="ISO-8859-1")
    df = df.dropna()
    
    # Ensure percentages are valid
    if train_percent + valid_percent >= 100:
        raise ValueError("The sum of train and validation percentages must be less than 100.")
    
    # Compute test percentage
    test_percent = 100 - (train_percent + valid_percent)
    
    # Split into train + valid and test first
    train_valid_data, test_data = train_test_split(
        df, test_size=test_percent / 100, random_state=42, shuffle=True
    )
    
    # Further split train + valid into train and valid
    train_data, valid_data = train_test_split(
        train_valid_data, test_size=valid_percent / (train_percent + valid_percent), random_state=42, shuffle=True
    )
    
    # Save the splits to separate CSV files
    train_data.to_csv(output_train, index=False, encoding="ISO-8859-1")
    valid_data.to_csv(output_valid, index=False, encoding="ISO-8859-1")
    test_data.to_csv(output_test, index=False, encoding="ISO-8859-1")
    
    print(f"Data split completed:")
    print(f"  Train: {len(train_data)} rows saved to {output_train}")
    print(f"  Validation: {len(valid_data)} rows saved to {output_valid}")
    print(f"  Test: {len(test_data)} rows saved to {output_test}")

# Main script
if __name__ == "__main__":
    # Ask user for inputs
    train_percent = float(input("Enter the percentage of data for training (0-100): "))
    valid_percent = float(input("Enter the percentage of data for validation (0-100): "))

    # Call the function
    split_data(train_percent, valid_percent)
