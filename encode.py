import pandas as pd
import category_encoders as ce


def encode_database(x0, y0, categorical_columns, save_path=None):
    """
    Encode categorical features in a pandas DataFrame using CatBoost encoder

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to encode
    categorical_columns : list or tuple
        List of column indices to treat as categorical features
    target_column : int
        Index of the target column (default: -1 for last column)
    save_path : str, optional
        Path to save the encoded DataFrame as xlsx file

    Returns:
    --------
    pd.DataFrame
        Encoded DataFrame
    """
    # Make a copy to avoid modifying the original DataFrame
    df = pd.concat([x0, y0], axis=1).copy()

    # Get column information
    columns = df.columns.tolist()
    # print(f"Dataset columns: {columns}")
    # print(f"Dataset shape: {df.shape}")

    # Separate features and target

    X = df.iloc[:, :-1]  # All columns except the last one
    y = df.iloc[:, -1]  # Last column
    y = y.astype(float)

    # Identify categorical features by column indices
    categorical_feature_names = [columns[i] for i in categorical_columns]
    # print(f"Categorical feature columns: {categorical_feature_names}")

    # Use CatBoost encoder
    cat_boost_encoder = ce.OrdinalEncoder(cols=categorical_feature_names)

    # Encode categorical features
    X_copy = X.copy()
    X_encoded = cat_boost_encoder.fit_transform(X_copy, y)
    X_encoded = X_encoded.astype(float)

    # Concatenate encoded features with target variable

    df_encoded = pd.concat([X_encoded, y], axis=1)

    # Keep original column names
    # df_encoded.columns = columns

    # Save to file if path provided
    if save_path:
        df_encoded.to_excel(save_path, index=False)
        # print(f"Encoded data saved to: {save_path}")

    # Display information
    # print("\nEncoding completed!")
    # print(f"Original data shape: {df.shape}")
    # print(f"Encoded data shape: {df_encoded.shape}")

    # Display before and after encoding comparison
    # print("\nOriginal data preview:")
    # print(df.head())

    # print("\nEncoded data preview:")
    # print(df_encoded.head())

    # # Display encoding information
    # print("\nCategorical feature encoding info:")
    for feature in categorical_feature_names:
        original_values = df[feature].unique()
        # print(f"{feature}: {len(original_values)} unique values")
        # print(
        #     f"  Original values: {list(original_values)[:10]}{'...' if len(original_values) > 10 else ''}"
        # )

    return X_encoded, cat_boost_encoder, categorical_feature_names


if __name__ == "__main__":
    # Example usage
    # Read data from Excel file
    df = pd.read_excel("database2.xlsx")

    # Encode first two columns as categorical features (columns 0 and 1)
    # Target column is the last column (-1)
    # Save result to database_encoded.xlsx
    encoded_df = encode_database(
        df=df,
        categorical_columns=[0, 1],
        target_column=-1,
        save_path="database_encoded.xlsx",
    )
